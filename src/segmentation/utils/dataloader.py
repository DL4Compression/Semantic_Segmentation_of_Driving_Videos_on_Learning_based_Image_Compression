import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import cv2
import os
from PIL import Image
from torch.utils import data
import numpy as np

class SegmentationDataset(data.Dataset):
    def __init__(self, path_img, path_mask,size,input_mode ,test=False, data_transforms=None):
        self.data_transforms = data_transforms
        self.path_img = path_img
        self.size = size
        self.path_mask = path_mask
        self.input_mode = input_mode
        list_img = []
        for itr in os.listdir(self.path_img):
            list2 = [itr+"/"+st for st in os.listdir(os.path.join(path_img,itr))]
            list_img = list_img + list2
        list_mask = []		
        for itr1 in os.listdir(self.path_mask):
            list3 = [itr1+"/"+st for st in os.listdir(os.path.join(path_mask,itr1))]
            list_mask = list_mask + list3
        self.seg_list = list_mask
        self.img_list = list_img
        self.test = test
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        if self.input_mode == 'latent':
            inp = np.load(os.path.join(self.path_img,self.img_list[index]))
            inp = inp[0]
            inp = np.transpose(inp,(1,2,0))
        else:
            inp = Image.open(os.path.join(self.path_img,self.img_list[index]))
        if(self.input_mode == 'latent'):
            trf = transforms.ToTensor()
            inp = trf(inp)
        else:
            inp = self.data_transforms(inp)
        seg = cv2.imread(os.path.join(self.path_mask, self.seg_list[index]))
        seg = torch.from_numpy(seg[:,:,0]).long()
        return inp, seg

class SegmentationDataset_test(data.Dataset):
    def __init__(self, path_img, path_mask,size,input_mode ,test=True, data_transforms=None):
        self.data_transforms = data_transforms
        self.path_img = path_img
        self.size = size
        self.path_mask = path_mask
        self.input_mode = input_mode
        self.seg_list = [os.path.join(self.path_mask,file) for file in os.listdir(self.path_mask)]
        self.img_list = [os.path.join(self.path_img,file) for file in os.listdir(self.path_img)]
        self.test = test
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        if self.input_mode == 'latent':
            inp = np.load(os.path.join(self.path_img,self.img_list[index]))
            inp = inp[0]
            inp = np.transpose(inp,(1,2,0))
        else:
            inp = Image.open(os.path.join(self.path_img,self.img_list[index]))
        if(self.input_mode == 'latent'):
            trf = transforms.ToTensor()
            inp = trf(inp)
        else:
            inp = self.data_transforms(inp)
        seg = cv2.imread(os.path.join(self.path_mask, self.seg_list[index]))
        seg = torch.from_numpy(seg[:,:,0]).long()
        name = self.img_list[index].split('/')[-1]
        if self.input_mode == 'latent':
            name = name.split('.')[0]
            name = name + '.png'
        return inp,seg,name
