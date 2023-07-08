#!/usr/bin/env bash

# train the net (suppose 8 gpus)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --data_set cityscapes \
--img_dir #add patyh to latent spaces of output of compressor of d=2 \
--lbl_dir #add the directory to the respective ground truths \
--arch DualSeg_res50 \
--input_size 832 \
--batch_size_per_gpu 3 \
--rgb 1 \
--learning_rate 0.01 \
--num_steps 50000 \
--save_dir "./" \
--ohem 1 --ohem_thres 0.7 --ohem_keep 100000 \
--log_file "dual_seg_r50_city.log" \
--restore_from "/home/pragyadipta/Semantic-Segmentation-of-Driving-Videos-on-Learning-based-Image-Compression/model_weights_icme/DualSeg_res50_final.pth"
