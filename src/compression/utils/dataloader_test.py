from torch.utils import data
from PIL import Image,ImageOps
import os, torchvision
import random
import torch
from torch import nn
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
# from torchmetrics import PeakSignalNoiseRatio
import numpy as np
class IDDCOM(data.Dataset):
	'''
	Specialized version of 'Dataset' for IDD-COMPRESSION dataset
	'''

	def __init__(self, path_to_dataset, 
				transform_images = None, transform_masks = None,
				images_path_rel = '.', masks_path_rel = '.',
				preserve_names = False):
		self.path_to_dataset = os.path.abspath(path_to_dataset) # root folder of the CBIS-DDSM dataset
		self.images_path_rel = images_path_rel # relative path to images
		self.masks_path_rel = masks_path_rel # relative path to masks (same as images)
		self.transform_images = transform_images # transforms
		self.transform_masks = transform_masks # transforms
		self.preserve_names = preserve_names # not important, debugging stuff

		# This is the list of all samples
		self.cropimages = os.listdir(os.path.join(self.path_to_dataset, self.images_path_rel))

		# choose random samples for one epoch
		self.choose_random_subset()

	def choose_random_subset(self, how_many = 0):
		# chooses 'how_many' number of samples and discard the rest
		if how_many == 0:
			self.cropsubset = self.cropimages
		else:
			self.cropsubset = random.sample(self.cropimages, how_many)

	def __len__(self):
		return len(self.cropsubset)

	def __getitem__(self, i):
		# indexing function

		if not hasattr(self, 'cropsubset'):
			# if not chosen a subset, randomly chose the default 'how_many'
			self.choose_random_subset()

		# Read the images and masks (same as images)
		image = Image.open(os.path.join(self.path_to_dataset, self.images_path_rel, self.cropsubset[i]))
		# image = ImageOps.grayscale(image)
		# print(image.shape)
		mask = Image.open(os.path.join(self.path_to_dataset, self.masks_path_rel, self.cropsubset[i]))
		# mask = ImageOps.grayscale(mask)

		# usual transformation apply
		if self.transform_images is not None:
			image = self.transform_images(image)
		if self.transform_masks is not None:
			mask = self.transform_masks(mask)
		# debugging stuff, not important
		if self.preserve_names:
			return image, mask, self.cropsubset[i]
		else:
			return image, mask


		
if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(
	"""
	Training script.

	Provide the folder paths which contains traning (patch) images and testing (full-scale mammogram) images produced by dataprep.py.
	Provide a filename prefix to help differentiate model/log from different runs with different configurations.

	(Optionally) You may choose to provide batch size (--batch_size), number of epochs (--epoch), printing interval (--interval).
	(Optionally) You may choose to limit the number of samples (--subset_size) to be used per epoch. '0' means the whole dataset.

	"""
	)
	parser.add_argument('--traindata', type=str, required=True, help='Folder path of the cropped images (train)')
	parser.add_argument('-b', '--batch_size', type=int, required=False, default=32, help='Batch size')
	parser.add_argument('-s', '--subset_size', type=int, required=False, default=0, help='Size of the randomly selected subset')

	args = parser.parse_args()

	main( args )
