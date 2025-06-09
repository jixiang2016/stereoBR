#!/usr/bin/env bash


### dataset_name: realBR (output_num=9), GOPRO-VFI_copy(output_num=7)
### Please update "input_dir", "dataset_name" and etc. 
### data_mode1='Blur', data_mode2 = 'RS', 'stereoRS'

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=10433 train.py \
        --input_dir='/media/zhongyi/D/data' \
        --dataset_name='realBR' \
		--output_dir='./train_log' \
		--output_num=9 --data_mode1='Blur' --data_mode2='RS' --epoch=1000 --batch_size=2  --batch_size_val=2 --prompt  --distilled  # --resume=True 


