
import os
import os.path as osp
import numpy as np
import random
import cv2

from torch.utils import data


class Cityscapes(data.Dataset):
    def __init__(self, img_dir, lbl_dir, max_iters=None, crop_size=(321, 321),
                 mean=(128, 128, 128), vars=(1,1,1), scale=True, mirror=True, ignore_label=255, RGB=False,eval=False):
        self.img_dir = img_dir
        self.eval = eval
        self.lbl_dir = lbl_dir
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.vars = vars
        self.is_mirror = mirror
        list_img = []
        for itr in os.listdir(self.img_dir):
            list2 = [os.path.join(img_dir,itr+"/"+st) for st in os.listdir(os.path.join(img_dir,itr))]
            list_img = list_img + list2
        list_mask = []		
        for itr1 in os.listdir(self.lbl_dir):
            list3 = [os.path.join(lbl_dir,itr1+"/"+st) for st in os.listdir(os.path.join(lbl_dir,itr1))]
            list_mask = list_mask + list3
        self.files = []
        print(list)
        for id in range(len(list_img)):
            # print(img_file)
            self.files.append({
                "img": list_img[id],
                "label": list_mask[id],
            })
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]

        img_pth = datafiles["img"]
        name = img_pth.split("/")[-1].split(".")[0]
        image = np.load(img_pth)
        img_list = img_pth.split("/")[-1].split("_")
        img_sub_dir = img_pth.split("/")[-2]
        fin = img_list[-1].split(".")[0].split("bit")[-1]

        lbl_path = img_list[0]+"_"+img_list[1]+"_"+img_list[2]+"_"+"gtFine"+"_labelIds"+fin+".png"

        label = cv2.imread(os.path.join(self.lbl_dir,img_sub_dir,lbl_path), cv2.IMREAD_GRAYSCALE)

        label = self.id2trainId(label)
        image = np.asarray(image, np.float32)[0]
        image -= self.mean[0]
        image /= self.vars[0]
        if self.eval:
            return image.copy(), label.copy(), name
        return image.copy(), label.copy()
