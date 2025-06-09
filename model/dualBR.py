import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.warplayer import warp
from model.stn import STN


def edge_detection(tensor, quantile_threshold=0.75):

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_x = sobel_x.repeat(3, 1, 1, 1)  # (3, 1, 3, 3)
    sobel_y = sobel_y.repeat(3, 1, 1, 1)  # (3, 1, 3, 3)
    sobel_x = sobel_x.to(tensor.device)
    sobel_y = sobel_y.to(tensor.device)

    grad_x = F.conv2d(tensor, sobel_x, padding=1, groups=3)  # (B, 3, H, W)
    grad_y = F.conv2d(tensor, sobel_y, padding=1, groups=3)  # (B, 3, H, W)
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # (B, 3, H, W)
    edge_mask = grad_magnitude.mean(dim=1, keepdim=True)  # (B, 1, H, W)

    min_val = edge_mask.amin(dim=(-2, -1), keepdim=True)  
    max_val = edge_mask.amax(dim=(-2, -1), keepdim=True)  
    normalized_edge_mask = (edge_mask - min_val) / (max_val - min_val + 1e-8)  

    flat_edge_mask = normalized_edge_mask.flatten(start_dim=-2)  
    threshold = torch.quantile(flat_edge_mask, quantile_threshold, dim=-1, keepdim=True)  # (B, 1, 1)
    
    binary_mask = (normalized_edge_mask > threshold.unsqueeze(-1)).float()  # (B, 1, H, W)

    return binary_mask  #normalized_edge_mask
    
    
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel)) #torch.nn.LayerNorm
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def warp_images(img, flows,length):
    warp_list = []
    for idx in range(length):
        flow_idx = flows[:, 2*idx:2*(idx+1)]
        warp_list.append(warp(img, flow_idx))
    warped_imgs = torch.cat(warp_list,1)
    return warped_imgs
    
    
class dualBR(nn.Module):
    def __init__(self,input_num,output_num, distilled,prompt):
        super(dualBR, self).__init__()
        self.input_num = input_num
        self.output_num = output_num         
        self.distilled = distilled
        self.prompt = prompt
        self.block0 = MotionIntrpl(3,c=240,out_imgs=self.output_num,\
                              in_planes_s=6)
        self.block1 = MotionIntrpl(3*(1+output_num)+output_num*2, c=150,out_imgs=self.output_num,\
                              in_planes_s=10+output_num)
        self.block2 = MotionIntrpl(3*(1+output_num)+output_num*2, c=90,out_imgs=self.output_num,\
                              in_planes_s=10+output_num)
        
        if self.distilled:
            self.block_tea = MotionIntrpl(3*(1+output_num)+output_num*2+3*output_num, c=90,out_imgs=self.output_num,\
                              in_planes_s=10+output_num+3*output_num)
        if self.prompt:
            self.pgm = PGM(output_num)
        
        #####  GenNet
        self.contextnet = Contextnet()
        self.unet = Unet(input_num, output_num)
        
    def forward(self, imgs_tensor,temp_map,gts_tensor=None, scale=[4,2,1],training=False):
    
        blur = imgs_tensor[:,:3]  
        rg = imgs_tensor[:,3:]  
       
        mask_list = [] 
        flow_list = [] 
        warped_imgs_list = []
        merged_warped_imgs_list = []  

        loss_distill = 0
        flow_teacher = None 
        merged_imgs_teacher = None 
        
        flow = None 
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:                   
                flow_d, mask_d, flow_br_d= stu[i](imgs_tensor,temp_map,None,warped_imgs,flow, mask, flow_br,scale=scale[i])                  
                flow = flow + flow_d
                mask = mask + mask_d
                flow_br = flow_br + flow_br_d
            else:
                flow, mask, flow_br = stu[i](imgs_tensor,temp_map,None, None,None,None,None,scale=scale[i]) 
            
            warped_imgs_b = warp_images(blur,flow[:,:2*self.output_num],self.output_num)  
            warped_imgs_r = warp_images(rg,flow[:,2*self.output_num:],self.output_num)  
            warped_imgs = torch.cat([warped_imgs_b, warped_imgs_r], 1)  
            warped_imgs_list.append(warped_imgs)
            flow_list.append(flow)
            mask_list.append(torch.sigmoid(mask))
        
        bb,_,hh,ww= imgs_tensor.shape
        
        if gts_tensor != None and self.distilled:
            flow_d, mask_d, flow_br_d= self.block_tea(imgs_tensor,temp_map,gts_tensor,warped_imgs,flow, mask, flow_br,scale=1)
            flow_teacher = flow + flow_d
            mask_teacher = torch.sigmoid(mask + mask_d).unsqueeze(2)  
            flow_br_teacher = flow_br + flow_br_d
            warped_imgs_teacher_b = warp_images(blur, flow_teacher[:, :2*self.output_num], self.output_num) 
            warped_imgs_teacher_r = warp_images(rg, flow_teacher[:, 2*self.output_num:], self.output_num)
            warped_imgs_teacher_b = warped_imgs_teacher_b.view(bb,self.output_num,3,hh,-1)
            warped_imgs_teacher_r = warped_imgs_teacher_r.view(bb,self.output_num,3,hh,-1)
   
            merged_imgs_teacher = warped_imgs_teacher_b * mask_teacher + (1.0-mask_teacher)*warped_imgs_teacher_r 

            # edge area   
            loss_mask_edge = edge_detection(gts_tensor.reshape(bb*self.output_num,3,hh,-1))  
            loss_mask_edge = loss_mask_edge.view(bb,self.output_num,1,hh,-1).float().detach()
            # motion magnitude (large motion area)
            flow_teacher_m = flow_teacher.view(bb,2,self.output_num,2,hh,-1)
            uu = flow_teacher_m[:,:,:,:1] 
            vv = flow_teacher_m[:,:,:,1:]
            magnitude = torch.sqrt(uu**2 + vv**2).sum(dim=1)
            Q1 = torch.quantile(magnitude.flatten(start_dim=-2), 0.25, dim= -1, keepdim=True)  
            Q3 = torch.quantile(magnitude.flatten(start_dim=-2), 0.75, dim= -1, keepdim=True) 
            adaptive_threshold = (Q3 + 2*(Q3-Q1)).unsqueeze(-1) 
            loss_mask_dyn = (magnitude > adaptive_threshold).float().detach() 
            
        for j in range(2,3):  
            mask_pred = mask_list[j].unsqueeze(2)

            warped_img_pred_t2b = warped_imgs_list[j][:,:3*self.output_num].view(bb,self.output_num,3,hh,-1)
            warped_img_pred_b2t = warped_imgs_list[j][:,3*self.output_num:].view(bb,self.output_num,3,hh,-1)
            merged_warped_imgs = warped_img_pred_t2b * mask_pred + (1.0-mask_pred)*warped_img_pred_b2t 
            merged_warped_imgs_list.append(merged_warped_imgs.view(bb,self.output_num*3,hh,-1))
            
            if gts_tensor != None and self.distilled: 
                #
                loss_mask_diff = ((merged_warped_imgs - gts_tensor.view(bb,self.output_num,3,hh,-1)).abs().mean(2, keepdim=True) \
                               > (merged_imgs_teacher - gts_tensor.view(bb,self.output_num,3,hh,-1)).abs().mean(2, True) + 0.01).float().detach()

                loss_mask = (loss_mask_diff + loss_mask_dyn + loss_mask_edge)/3  
                
                flow_diff = (flow_teacher.detach() - flow_list[j]).view(bb,2,self.output_num,2,hh,-1)
                flow_diff = flow_diff.permute(0,2,1,3,4,5).reshape(bb,self.output_num,4,hh,-1) 
                loss_distill += (flow_diff.abs() * loss_mask).mean() # L1

            
      
        ### refinement
        c0 = self.contextnet(blur, flow[:,:2*self.output_num]) 
        c1 = self.contextnet(rg, flow[:,2*self.output_num:])

        wc0=[]
        wc1=[]
        ##### self-motion prompt 
        if self.prompt:
            motion_residue = flow[:,:2*self.output_num] - flow[:,2*self.output_num:]
            pt = self.pgm(motion_residue)
            for idx in range(len(c0)):
                wc0.append(c0[idx]+c0[idx]*torch.sigmoid(pt[idx])) 
                wc1.append(c1[idx]+c1[idx]*torch.sigmoid(pt[idx]))       
        else:
            wc0=c0
            wc1=c1
           
           
        tmp = self.unet(imgs_tensor, warped_imgs, flow, mask, wc0, wc1)
        
        res = tmp[:,:3*self.output_num] * 2 - 1 
        final_out = torch.clamp(merged_warped_imgs_list[-1] + res, 0, 1) 
         
        if gts_tensor != None and self.distilled:
            merged_imgs_teacher = merged_imgs_teacher.view(bb,self.output_num*3,hh,-1)
        return flow_list, warped_imgs_list,final_out, merged_imgs_teacher,flow_teacher,loss_distill

class MotionIntrpl(nn.Module):
    def __init__(self, in_planes, c, out_imgs,in_planes_s):
        super(MotionIntrpl, self).__init__()
        self.output_num = out_imgs
        out_planes = 2*self.output_num
        
        
        #### contextual branch for blur input
        self.conv0_b = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock_b = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),  #### (c,c/2)
        )
        self.lastconv_b = nn.ConvTranspose2d(2*c+c, out_planes, 4, 2, 1)  ###(c+c/3)
        
        
        
        #### temporal branch for RS/RSGR input
        self.conv0_r = nn.Sequential(
            conv(in_planes+self.output_num, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock_r = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),  #### (c,c/2)
        )
        self.lastconv_r = nn.ConvTranspose2d(2*c+c, out_planes, 4, 2, 1)  ###(c+c/3)

        
        
        ### shutter alignment
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_planes_s, 64, 3, 1, 1),
            ResBlock(64, 64),
            ResBlock(64, 64),
            conv(64,128,3,2,1),
            )   
        self.stn_br_1 = STN(in_channels = 128)
        self.stn_rb_1 = STN(in_channels = 128)
        self.refine1 = nn.Sequential(
            conv(128*2,256),
            nn.Conv2d(256, 128, 3, 1, 1)
            ) 
                      
        self.encoder2 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            conv(128,256,3,2,1),
            )
        self.stn_br_2 = STN(in_channels = 256)
        self.stn_rb_2 = STN(in_channels = 256)
        self.refine2 = nn.Sequential(
            conv(128+256*2,256),
            nn.Conv2d(256, 2, 3, 1, 1)
            )  
        
        self.conv_e = nn.Conv2d(256, c, 3, 1, 1)
        self.out = nn.Conv2d(c, 1*out_imgs, 3, 1, 1)
        
        
    def forward(self, imgs_tensor,temp_map,gts_tensor, warped_imgs,flow, mask, flow_br,scale):
        #temp_map (b,output_num,h,w)
        height,width = imgs_tensor.shape[-2:]
        imgs_tensor = F.interpolate(imgs_tensor, scale_factor = 1. / scale, mode="bilinear", align_corners=False)  # resize to 1/K == 1/4,2,1
        temp_map = F.interpolate(temp_map, scale_factor = 1. / scale, mode="bilinear", align_corners=False) 
        blur = imgs_tensor[:,:3]
        RG = torch.cat([imgs_tensor[:,3:],temp_map],dim=1)
        #temp_map_m = F.interpolate(temp_map, scale_factor = 1. / 2, mode="bilinear", align_corners=False)
        #### Teacher Network
        if gts_tensor != None:  ### (B, 3*output_num,H,W)
            gts_tensor = F.interpolate(gts_tensor, scale_factor = 1. / scale, mode="bilinear", align_corners=False)  # resize to 1/K == 1/4,2,1
            
            blur = torch.cat((blur, gts_tensor), 1)
            RG = torch.cat((RG, gts_tensor), 1)
            imgs_tensor = torch.cat((imgs_tensor, gts_tensor), 1)  ### need to keep ?????
        if flow != None:
            warped_imgs = F.interpolate(warped_imgs, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            flow_br = F.interpolate(flow_br, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            
            imgs_tensor = torch.cat((imgs_tensor, flow_br,mask), 1)
            blur = torch.cat((blur, flow[:,:self.output_num*2],warped_imgs[:,:self.output_num*3]), 1)
            RG = torch.cat((RG, flow[:,self.output_num*2:],warped_imgs[:,self.output_num*3:]), 1)
            
        ### shutter alignment
        out1 = self.encoder1(imgs_tensor)##(B,128,h/2,w/2)
        flow_b2r_1 = self.refine1(torch.cat([out1,self.stn_br_1(out1)],dim=1))#(B,128,h/2,w/2)
        flow_b2r_1 = F.interpolate(flow_b2r_1, scale_factor = 1. / 2, mode="bilinear", align_corners=False)
        flow_r2b_1 = self.refine1(torch.cat([out1,self.stn_rb_1(out1)],dim=1))#(B,128,h/2,w/2)
        flow_r2b_1 = F.interpolate(flow_r2b_1, scale_factor = 1. / 2, mode="bilinear", align_corners=False)
        out2 = self.encoder2(out1)##(B,256,h/4,w/4)
        flow_b2r_2 = self.refine2(torch.cat([out2,self.stn_br_2(out2),flow_b2r_1],dim=1))#(B,2,h/4,w/4)
        flow_r2b_2 = self.refine2(torch.cat([out2,self.stn_rb_2(out2),flow_r2b_1],dim=1))#(B,2,h/4,w/4)
        ext_fea = self.conv_e(out2) # ##(B,c//3,h/4,w/4)
        mask = self.out(ext_fea) ###(B,output_num,h/4,w/4)
        mask = F.interpolate(mask, size = [height,width], mode="bilinear", align_corners=False)###(B,output_num,h,w)
        
        
        ## blur branch        
        x = self.conv0_b(blur)
        x_b = self.convblock_b(x) + x
        
        ## RG branch        
        x = self.conv0_r(RG)
        x_r = self.convblock_r(x) + x ##(B,c,h/4,w/4)
        
        ## blur branch 
        x_b_w = warp(x_r,flow_b2r_2)
        tmp_b = self.lastconv_b(torch.cat([x_b,x_b_w,ext_fea],dim=1))
        #tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        tmp_b = F.interpolate(tmp_b, size = [height,width], mode="bilinear", align_corners=False)
        # flow_t2b: (batch_size,output_num*2,h,w)
        flow_t2b = tmp_b[:,:2*self.output_num] * scale * 2
        
        ## RG branch
        x_r_w = warp(x_b,flow_r2b_2)
        tmp_r = self.lastconv_r(torch.cat([x_r,x_r_w,ext_fea],dim=1))
        #tmp * temp_map_m
        #tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        tmp_r = F.interpolate(tmp_r, size = [height,width], mode="bilinear", align_corners=False)
        # flow_t2r: (batch_size,output_num*2,h,w)
        flow_t2r = tmp_r[:,:2*self.output_num] * scale * 2
        
        flow = torch.cat([flow_t2b,flow_t2r],dim=1)  # [flow_t2b,flow_t2r] (batch_size,output_num*4,h,w)
        
        flow_br = torch.cat([flow_b2r_2,flow_r2b_2],dim=1) # [flow_b2r,flow_r2b] (batch_size,4,h,w)
        flow_br = F.interpolate(flow_br, size = [height,width], mode="bilinear", align_corners=False)
        flow_br = flow_br * scale * 4

        return flow, mask, flow_br

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        c = 16
        self.conv1 = Conv2(3, c//2)
        self.conv2 = Conv2(c//2, c)
        self.conv3 = Conv2(c, 2*c)
        self.conv4 = Conv2(2*c, 4*c)
    
    def forward(self, x, flow):
        #x: tensor(batch_size, 3, h, w)  flow: tensor (batch_size,2*output_num,h,w)
        if flow != None:
            length = flow.shape[1]//2

        x = self.conv1(x)
        if (flow != None):
            size_list = [math.ceil(x/2) for x in flow.shape[-2:]]
            flow = F.interpolate(flow, size=size_list, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            #flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            f1 = warp_images(x,flow,length) # (batch_size,c/2*output_num,h,w)
        else:
            f1 = x.clone()  # (batch_size,c/2,h,w)       
        
        x = self.conv2(x)
        if (flow != None):
            size_list = [math.ceil(x/2) for x in flow.shape[-2:]]
            flow = F.interpolate(flow, size=size_list, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            #flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            f2 = warp_images(x,flow,length)# (batch_size,c*output_num,h,w)
        else:
            f2 = x.clone() # (batch_size,c,h,w) 
        
        x = self.conv3(x)
        if (flow != None):
            size_list = [math.ceil(x/2) for x in flow.shape[-2:]]
            flow = F.interpolate(flow, size=size_list, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            #flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            f3 = warp_images(x,flow,length)# (batch_size,2c*output_num,h,w)
        else:
            f3 = x.clone() # (batch_size,2c,h,w)
                
        x = self.conv4(x)
        if (flow != None):
            size_list = [math.ceil(x/2) for x in flow.shape[-2:]]
            flow = F.interpolate(flow, size=size_list, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            #flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
            f4 = warp_images(x,flow,length)# (batch_size,4c*output_num,h,w)
        else:
            f4 = x.clone() # (batch_size,4c,h,w)
        
        return [f1, f2, f3, f4]

### Prompt Generation Module   
class PGM(nn.Module):
    def __init__(self, output_num):
        super(PGM, self).__init__()
        self.output_num = output_num
        c = 16*output_num
        self.conv1 = Conv2(output_num*2, c//2)
        self.conv2 = Conv2(c//2, c)
        self.conv3 = Conv2(c, 2*c)
        self.conv4 = Conv2(2*c, 4*c)
    
    def forward(self, x):
        #x: tensor(batch_size, 2*output_num, h, w)
        x = self.conv1(x)
        p1 = x.clone()  # (batch_size,c/2,h/2,w/2)       
        
        x = self.conv2(x)
        p2 = x.clone() # (batch_size,c,h/4,w/4) 
        
        x = self.conv3(x)
        p3 = x.clone() # (batch_size,2c,h/8,w/8)
                
        x = self.conv4(x)
        p4 = x.clone() # (batch_size,4c,h/16,w/16)
        
        return [p1, p2, p3, p4]



class Unet(nn.Module):
    def __init__(self,input_num,output_num):
        super(Unet, self).__init__()
        self.input_num = input_num
        self.output_num = output_num
        out = output_num
        c = 16

        self.down0 = Conv2(5*out*2+out+3*input_num, 4*c) #  4c              (c/2*out)*2   
        self.down1 = Conv2(4*c+c*out, 4*c+c*out) #4*c+c*out  (c*out)*2 
        self.down2 = Conv2(4*c+3*c*out, 4*c+3*c*out) #4*c+3*c*out  (2c*out)2   
        self.down3 = Conv2(4*c+7*c*out, 4*c+7*c*out)#4*c+7*c*out  (4c*out)*2
        self.up0 = deconv(4*c+15*c*out, 4*c+3*c*out)#4*c+3*c*out  4*c+3*c*out
        self.up1 = deconv(8*c+6*c*out, 4*c+c*out)#4*c+c*out  4*c+c*out
        self.up2 = deconv(8*c+2*c*out, 4*c)#4*c 4*c
        self.up3 = deconv(8*c, 4*c)
        self.conv = nn.Conv2d(4*c, 3*out, 3, 1, 1)
        
        
    def forward(self,imgs_tensor, warped_imgs, flow, mask,c0, c1):
        # c2: pre c1:nxt
        s0 = self.down0(torch.cat((imgs_tensor, warped_imgs,flow,mask), 1))

        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))

        #!!!
        if x.shape[-2:] != s2.shape[-2:]:
            x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat((x, s2), 1)) 
        #!!!
        if x.shape[-2:] != s1.shape[-2:]:
            x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat((x, s1), 1)) 
        #!!!
        if x.shape[-2:] != s0.shape[-2:]:
            x = F.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat((x, s0), 1)) 
        
        x = self.conv(x)  ##(batch, 3*output_num, h, w)
        
        return torch.sigmoid(x)
        
        
        
