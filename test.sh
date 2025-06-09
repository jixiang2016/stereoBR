#!/usr/bin/env b
# For testing on benchmark dataset

### dataset_name: realBR (output_num=9), GOPRO-VFI_copy(output_num=7)
### Please update "input_dir", "dataset_name" and etc.
### data_mode1='Blur', data_mode2 = 'RS', 'stereoRS'


CUDA_VISIBLE_DEVICES=0 python3 test.py \
        --input_dir='/media/zhongyi/D/data' \
        --dataset_name='realBR' \
		--output_dir='./output' \
		--output_num=9   \
		--model_dir='./train_log/realBR_Blur-RS_2_9/best.ckpt' \
		--data_mode1='Blur'  --data_mode2='RS'  --keep_frames  --batch_size=3 --prompt --distilled #--temporal --keep_flows

