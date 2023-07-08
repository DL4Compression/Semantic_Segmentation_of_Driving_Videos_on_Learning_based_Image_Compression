CUDA_VISIBLE_DEVICES=1 python infer_and_save.py --data_set cityscapes \
--img_dir "/storage/ravi/test_set_5/lat_5_d2" \
--lbl_dir "/storage/ravi/test_set_5/gt" \
--arch DualSeg_res50 \
--rgb 1 \
--restore_from "/home/pragyadipta/Semantic-Segmentation-of-Driving-Videos-on-Learning-based-Image-Compression/model_weights_icme/DualSeg_res50_final_d2.pth" \
--output_dir "./test_d2_new_5" \
--num_classes 19 \
--whole 1 
# --img_dir /storage/ravik/jpg_vic \