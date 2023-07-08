CUDA_VISIBLE_DEVICES=1 python infer_and_save.py --data_set cityscapes \
--img_dir "#add path to the saved latent vector spaces of d=2 model" \
--lbl_dir "# add path to the ground truths" \
--arch DualSeg_res50 \
--rgb 1 \
--restore_from " add path to the model that you have downloaded named as DualSeg_res50_final_d2.pth or your trained model" \
--output_dir "#add path to output directory" \
--num_classes 19 \
--whole 1 
