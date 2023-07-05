import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm as tq
from skimage.util import view_as_windows


def main (args):
    folder_path = args.path_to_dataset
    folder_list = os.listdir(folder_path)
    for idx1,sub_fold in enumerate(tq(os.listdir(folder_path))):
        if not os.path.exists(os.path.join(args.out_path,sub_fold)):
            os.mkdir(os.path.join(args.out_path,sub_fold))
        image_list = [os.path.join(folder_path,sub_fold,image_name) for image_name in os.listdir(os.path.join(folder_path,sub_fold))]
        for idx,image in enumerate(tq(image_list)):
            img = Image.open(image)
            file_name = image.split('/')[-1].split('.')[0]
            ext = image.split('/')[-1].split('.')[-1]
            imv = np.array(img)
            imag = imv
            if imag.shape[0] > args.patch_dim and imag.shape[1] > args.patch_dim:
                if(len(imag.shape)==3):
                    patches = view_as_windows(imag, window_shape=(args.patch_dim, args.patch_dim,3), step=args.step)
                    for chid, ch in enumerate(patches):
                        for rowid, row in enumerate(ch):
                            for colid, patch in enumerate(row):
                                npim = patch.astype(np.uint8)
                                imsave = Image.fromarray((patch).astype(np.uint8))
                                save_folder_path = args.out_path
                                savepath = os.path.join(save_folder_path,sub_fold,file_name)+str(chid)+str(rowid)+'.'+ext
                                imsave.save(savepath)
                else:
                    patches = view_as_windows(imag, window_shape=(args.patch_dim, args.patch_dim), step=args.step)
                    for chid, ch in enumerate(patches):
                        for rowid, patch in enumerate(ch):
                            npim = patch.astype(np.uint8)
                            imsave = Image.fromarray((patch).astype(np.uint8))
                            save_folder_path = args.out_path
                            savepath = os.path.join(save_folder_path,sub_fold,file_name)+str(chid)+str(rowid)+'.'+ext
                            imsave.save(savepath)
        


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(
	"""
	Data preperation script. It may take a little while to generate all patches and CT. You can interrupt the generation in the middle by CTRL+C.

	Provide the folder path of original CBIS-DDSM dataset. This folder has to contain folders named 'Calc-Test_P_00127_RIGHT_MLO' etc.
	Provide two directories (--out_train and --out_test) to store trainig images (patches) and testing images (full-scale mammograms).

	(Optionally) Provide the patch size (--patch_dim). We strongly recommend the default value (256).
	"""
	)

	parser.add_argument('-p', '--path_to_dataset', type=str, required=True, help='Path to the original CBIS-DDSM folder')
	parser.add_argument('-d', '--patch_dim', type=int, required=False, default=256, help='Patches of size (patch_dim x patch_dim)')
	parser.add_argument('-o','--out_path', type=str, required=True, help='Output folder for dataset samples')
	parser.add_argument('-s', '--step', type=int, required=False, default=256, help='Patches of size (patch_dim x patch_dim)')

	args = parser.parse_args()
	main( args )