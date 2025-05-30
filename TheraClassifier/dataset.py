#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:23:52 2025

@author: alfonso
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd

#Class definition for reading data during training. 
#The argument 'transform' is applied to every input image no matter if current
#stage is training, validation or testing. However the  'train_transform' argument 
#contains transformations for data augmentation applied only during stage=train  
class TrainDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None,
                 train_transform=None, stage=''):
        self.images_dir = images_dir
        self.transform = transform
        self.train_transform=train_transform
        self.stage=stage        
        #I like to have my data organized in a pandas dataframe
        #then I create a pd column with image files. Filenames are formatted accordingly
        self.df = pd.read_csv(labels_file,header=None, names=['cohort'])
        files = []
        #Format filenames accordinlgy
        K='0'
        for r in range(1,self.df.cohort.size+1): 
            file = str(r)
            N = len(str(self.df.cohort.size))-len(file)
            file=K*N+file+'.jpg'
            files.append(file)
        self.df ['filename'] = files        
        #debug slice dataframe to run in debugging mode
        #self.df = self.df[:1000]
        
        self.image_paths = self.df['filename'].tolist()
        self.labels = self.df['cohort'].tolist()
    #Modify the pipeline transformatios applied only to training data 
    def modifyTransform(self, train_transform):
        self.stage='train'
        self.transform=train_transform
        self.train_transform=train_transform
        
    #Creates a subset given by the indexes subset_idx     
    def subset(self, subset_idx):
        self.df=self.df.loc[subset_idx].reset_index(drop=True)
        self.image_paths = self.df['filename'].tolist()
        self.labels = self.df['cohort'].tolist()       
        
    def __len__(self):
        return len(self.image_paths)
    #Used by the DataLoader
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, transform):
        self.images_dir = images_dir
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(images_dir) if '.jpg' in f])
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image

def get_test_dataloader(images_dir, transform, batch_size=32,num_workers=8):
    dataset = TestDataset(images_dir, transform)
    image_names = dataset.image_names
    #Dataloaders for reading testing samples
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader, image_names