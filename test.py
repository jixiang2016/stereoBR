import os
import sys
import cv2
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.trainer import Model
from dataset.GoproDualDataset import *
from dataset.DualRealDataset import *
from lpips import lpips
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
parser = argparse.ArgumentParser()

parser.add_argument('--training', default=False,  type=bool)
parser.add_argument('--input_num', default=2, type=int, help='input images number')
parser.add_argument('--input_dir', default='/media/zhongyi/D/data/GOPRO_RSGR',  type=str, required=True, help='path to the input dataset folder')
parser.add_argument('--dataset_name', default='realBR',  type=str, required=True, help='Name of dataset to be used')
parser.add_argument('--data_mode1', default='Blur',  type=str, help='Mode of input data')
parser.add_argument('--data_mode2', default='RS',  type=str, help='Mode of input data')
parser.add_argument('--temporal', action='store_true',default=False)
parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
parser.add_argument('--output_num', default=5, type=int, help='final output channel of the network')
parser.add_argument('--output_dir', default='',  type=str, required=True, help='path to save testing output')
parser.add_argument('--model_dir', default="./train_log/GOPROBase_RSGR_3_5/best.ckpt",  
                    type=str, help='path to the pretrained model folder')
parser.add_argument('--keep_frames', action='store_true', default=False, help='save interpolated frames')
parser.add_argument('--keep_flows', action='store_true', default=False, help='save predicted flows')
parser.add_argument('--distilled', action='store_true',default=False) 
parser.add_argument('--prompt', action='store_true',default=False) 
args = parser.parse_args()



def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return img

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def test(model): 
    model.load_model(path=args.model_dir)
    data_root = os.path.join(args.input_dir, args.dataset_name)
    
    
    if args.dataset_name == 'realBR':
        args.inter_num = 16   #16
        args.intra_num = 9  # 9 
        dataset_val = DualRealDataset(dataset_cls='test',\
                               input_num=args.input_num,\
                               output_num=args.output_num,\
                               data_root=data_root,\
                               data_mode1 = args.data_mode1,\
                               data_mode2 = args.data_mode2,\
                               inter_num = args.inter_num,\
                               intra_num = args.intra_num, temp=args.temporal) 
    elif args.dataset_name == 'GOPRO-VFI_copy':
        if args.output_num >8:
            raise Exception('Wrong output number!!!')
        args.inter_num = 0
        args.intra_num = 8 #8
        dataset_val = DualRealDataset(dataset_cls='test',\
                               input_num=args.input_num,\
                               output_num=args.output_num,\
                               data_root=data_root,\
                               data_mode1 = args.data_mode1,\
                               data_mode2 = args.data_mode2,\
                               inter_num = args.inter_num,\
                               intra_num = args.intra_num, temp=args.temporal) 
        
    elif args.dataset_name == 'GOPRO-Dual':   #### DeprecationWaring, dataset has been deleted.
        dataset_val = GoproDualDataset(dataset_cls='test',\
                               input_num=args.input_num,\
                               output_num=args.output_num,\
                               data_root=data_root)
    else:
        raise Exception('Not supported data!!!')

    val_data = DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=8)
    
    
    psnr_list = []
    psnr_dict = {}
    ssim_list = []
    ssim_dict = {}
    lpips_list = []
    lpips_dict = {}
    
    psnr_time = {}
    ssim_time = {}
    lpips_time = {}
    
    
    
    for i, all_data in enumerate(tqdm(val_data)):

        data = all_data[0]
        img_ids = all_data[1]
        gt_ids = np.array(all_data[2]).T
       
        data_gpu = data.to(device, non_blocking=True) / 255.
        imgs_tensor = data_gpu[:, :3*args.input_num] 
        gts_tensor = data_gpu[:, 3*args.input_num:]  

        
        ##### temporal-order encoding 
        batch,_,height,width = imgs_tensor.shape
        rs_encode = torch.arange(0,height).type_as(imgs_tensor).unsqueeze(1).repeat(1,width) ##(h,w)
        latent_gs_encode = []
        for out_i in range(0,args.output_num):
            gs_encode = torch.Tensor([(height-1)//(args.output_num-1)*out_i]).type_as(imgs_tensor).unsqueeze(0).repeat(height,width) #(h,w)
            latent_gs_encode.append(gs_encode)
        latent_gs_encodes = torch.stack(latent_gs_encode,dim=0)  
        ### relative location of ith latent gs to input rs
        latent_gs_encodes = rs_encode.unsqueeze(0) - latent_gs_encodes 
        latent_gs_encodes = latent_gs_encodes.unsqueeze(0).repeat(batch,1,1,1)
        
        
        with torch.no_grad():
            preds, flow_list,warped_imgs_list = model.inference(imgs_tensor,latent_gs_encodes)
        flows = flow_list[2]  
        warped_imgs = warped_imgs_list[2]
        
        batch_size = imgs_tensor.shape[0]
        for b_id in range(batch_size):
            pred = preds[b_id] 
            gt_tensor = gts_tensor[b_id] 
            flow = flows[b_id]
            warped_img = warped_imgs[b_id]
            gt_id = gt_ids[b_id] 
            img_id = img_ids[b_id] 
            
            seq_name = img_id.split('/')[1]
            img_name = img_id.split('/')[3]
            save_path = args.output_dir+'/'+args.dataset_name+'/'+seq_name+'/'+img_name
            if args.keep_flows is True or args.keep_frames is True:
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
            
            flow_t2b = flow[:2*args.output_num]
            flow_b2t = flow[args.output_num*2:]
            warped_img_t2b = warped_img[:3*args.output_num]
            warped_img_b2t = warped_img[3*args.output_num:]
            
            for o_id in range(args.output_num):
                p_img = pred[3*o_id:3*(o_id+1)] 
                g_img = gt_tensor[3*o_id:3*(o_id+1)]
                
                f_img_t2b = flow_t2b[2*o_id:2*(o_id+1)]
                f_img_b2t = flow_b2t[2*o_id:2*(o_id+1)]
                w_img_t2b = warped_img_t2b[3*o_id:3*(o_id+1)] 
                w_img_b2t = warped_img_b2t[3*o_id:3*(o_id+1)]
                
                g_id = gt_id[o_id] 
                ssim = ssim_matlab(g_img.unsqueeze(0),p_img.unsqueeze(0)).cpu().numpy()
                
                MAX_DIFF = 1 
                mse = torch.mean((g_img - p_img) * (g_img - p_img)).cpu().data
                psnr = 10* math.log10( MAX_DIFF**2 / mse )
 
                lpips=loss_fn_alex(p_img, g_img).cpu().item()

                if args.keep_frames is True:
                    gt_name = g_id.split('/')[-1]+'.png'
                    p_img_s = (p_img.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
                    cv2.imwrite(os.path.join(save_path,gt_name),p_img_s)
                if args.keep_flows is True:
                    gt_name_t2b = g_id.split('/')[-1]+'_flow_t2b.png'
                    f_img_s_t2b = (flow2rgb(f_img_t2b.permute(1,2,0).cpu().numpy()[:,:,::-1])*255).astype('uint8')  
                    cv2.imwrite(os.path.join(save_path,gt_name_t2b),f_img_s_t2b)
                    gt_name_b2t = g_id.split('/')[-1]+'_flow_b2t.png'
                    f_img_s_b2t = (flow2rgb(f_img_b2t.permute(1,2,0).cpu().numpy()[:,:,::-1])*255).astype('uint8')  
                    cv2.imwrite(os.path.join(save_path,gt_name_b2t),f_img_s_b2t)
                
                
                time_id = o_id+1
                if time_id not in lpips_time:
                    lpips_time[time_id] = []
                lpips_time[time_id].append(lpips)
                if time_id not in ssim_time:
                    ssim_time[time_id] = []
                ssim_time[time_id].append(ssim)
                if time_id not in psnr_time:
                    psnr_time[time_id] = []
                psnr_time[time_id].append(psnr)
                
                if seq_name not in lpips_dict:
                    lpips_dict[seq_name]={}
                if img_name not in lpips_dict[seq_name]:
                    lpips_dict[seq_name][img_name] = {}
                lpips_dict[seq_name][img_name][g_id.split('/')[-1]]= format(lpips,'.4f')
                lpips_list.append(lpips)
                
                if seq_name not in psnr_dict:
                    psnr_dict[seq_name]={}
                if img_name not in psnr_dict[seq_name]:
                    psnr_dict[seq_name][img_name] = {}
                psnr_dict[seq_name][img_name][g_id.split('/')[-1]]=format(psnr,'.4f')
                psnr_list.append(psnr)
                
                if seq_name not in ssim_dict:
                    ssim_dict[seq_name]={}
                if img_name not in ssim_dict[seq_name]:
                    ssim_dict[seq_name][img_name] = {}
                ssim_dict[seq_name][img_name][g_id.split('/')[-1]]=format(ssim,'.4f')
                ssim_list.append(ssim)
        
    
    save_dir = args.output_dir+'/'+args.dataset_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # keep txt record
    for seq_name,img_dict in psnr_dict.items():
        with open(save_dir+'/'+seq_name+'.txt','w') as f:
            for img_l in sorted(img_dict.items(),key=lambda x:x[0]):
                if args.dataset_name=='realBR' or args.dataset_name=='GOPRO-VFI_copy':
                    gt_dict_psnr = sorted(img_l[1].items(),key=lambda x:x[0])
                    gt_dict_ssim = sorted(ssim_dict[seq_name][img_l[0]].items(),key=lambda x:x[0])
                    gt_dict_lpips = sorted(lpips_dict[seq_name][img_l[0]].items(),key=lambda x:x[0])
                    for xx, yy, zz in zip(gt_dict_psnr, gt_dict_ssim,gt_dict_lpips):
                        assert xx[0] == yy[0] == zz[0]
                        f.write( xx[0]+'\t'+ xx[1]+'\t'+ yy[1]+'\t'+zz[1]+'\n')
                    
                elif args.dataset_name == 'GOPRO-Dual':
                    img_name = img_l[0]
                    for gt_l in sorted(img_l[1].items(),key=lambda x:x[0]):
                        img_psnr_record = gt_l[1]
                        img_ssim_record = ssim_dict[seq_name][img_name][gt_l[0]]
                        img_lpips_record = lpips_dict[seq_name][img_name][gt_l[0]]
                        f.write(img_name+'\t'+gt_l[0]+'\t'+img_psnr_record+'\t'+img_ssim_record+'\t'+img_lpips_record+'\n')
                else:
                    raise Exception('Not supported data!!!')
    
    with open(save_dir+'/overall_metrics.txt','w') as f:
        f.write('Overall PSNR: %.4f\n'%(np.mean(psnr_list)))
        f.write('Overall SSIM: %.4f\n'%(np.mean(ssim_list)))
        f.write('Overall LPIPS: %.4f\n'%(np.mean(lpips_list)))
        
        f.write('metrics by time stamp:\n')
        psnr_time = sorted(psnr_time.items(),key=lambda x:float(x[0]))
        for kk in psnr_time:
            avg_psnr = format(np.mean(kk[1]),'.4f')
            avg_ssim =  format(np.mean(ssim_time[kk[0]]),'.4f') 
            avg_lpips =  format(np.mean(lpips_time[kk[0]]),'.4f')
            f.write( 'tiemstamp:'+str(kk[0])+'\t'+ avg_psnr+'\t'+ avg_ssim+'\t'+avg_lpips+'\n')
    print('---------------------------------------------------------------')
    print('Overall PSNR: %.4f'%(np.mean(psnr_list)))
    print('Overall SSIM: %.4f'%(np.mean(ssim_list)))
    print('Overall LPIPS: %.4f'%(np.mean(lpips_list)))
    print('---------------------------------------------------------------')
   


if __name__ == "__main__":    
       
    # For reproduction 
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    model = Model(config=args)
    
    test(model)
        




