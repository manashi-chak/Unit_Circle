#!/usr/bin/env python
from __future__ import print_function, division
import collections
import os

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import torchvision.transforms
import pdb
import getch

import skimage
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
  

class IrisLoader(data.Dataset):

    def __init__(self, root, image_list_file, nonclass_id_label_dict, split='train', transform=True,horizontal_flip=False, upper=None):

        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.split = split
        self._transform = True
        self.nonclass_id_label_dict = nonclass_id_label_dict
        self.horizontal_flip = False
  
        self.img_info = []

        with open(self.image_list_file, 'r') as f:
            for i, img_files in enumerate(f):
                img_files = img_files.strip()  # e.g. train/n004332/0317_01.jpg
                img_file1 = img_files

                class1 = img_file1.split('/')[2]

                label = self.nonclass_id_label_dict[class1]    
                label=int(label)
                self.img_info.append({
                    'cid': class1,
                    'img1': img_file1,
                    'lbl': label,
                }) 
                if i % 1000 == 0:
                    print("processing: {} images for {}".format(i, self.split))
                if upper and i == upper - 1:  # for debug purpose
                    break   

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):

        info = self.img_info[index]
        img_file1 = info['img1']

        
        img1 = PIL.Image.open(os.path.join(self.root, img_file1))
        img1 = img1.resize((512, 110)) 
        img1 = np.array(img1, dtype=np.float32)
        img1 = np.divide(img1,(255*1.0))

        label = info['lbl']
        class_id = info['cid']
        return img1,  label, img_file1, class_id