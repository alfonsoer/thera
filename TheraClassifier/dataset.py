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

class BinaryImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # df = pd.read_csv(labels_file)
        df = pd.read_csv(labels_file,header=None, names=['cohort'])
        files = []
        K='0'
        for r in range(1,df.cohort.size+1): 
            file = str(r)
            N = len(str(df.cohort.size))-len(file)
            file=K*N+file+'.jpg'
            files.append(file)
        df ['filename'] = files        
        
        self.image_paths = df['filename'].tolist()
        self.labels = df['cohort'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image

def get_test_dataloader(image_dir, transform, batch_size=32):
    dataset = TestDataset(image_dir, transform)
    image_names = dataset.image_names
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, image_names