from torch.utils import data
from PIL import Image
import torch
import os

class IDD(data.Dataset):
	'''
	Specialized version of 'Dataset' for KLIV dataset
	'''

	def __init__(self, path_to_dataset, 
				transform_images = None):
		self.path_to_dataset = os.path.abspath(path_to_dataset) # root folder of the CBIS-DDSM dataset
		self.transform_images = transform_images # transforms
		list_img = []
		for itr in os.listdir(self.path_to_dataset):
			list2 = [itr+"/"+st for st in os.listdir(os.path.join(self.path_to_dataset,itr))]
			list_img = list_img + list2
		self.listimg = list_img

	def __len__(self):
		return len(self.listimg)

	def __getitem__(self, i):

		# Read the images and masks (same as images)
		image = Image.open(os.path.join(self.path_to_dataset,self.listimg[i]))
		

		# usual transformation apply
		if self.transform_images is not None:
			image = self.transform_images(image)
		# debugging stuff, not important
		return image

