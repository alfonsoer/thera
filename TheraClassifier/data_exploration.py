#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:03:16 2025

@author: alfonso
"""

import os
import pandas as pd
from PIL import Image
from dataset import TrainDataset, TestDataset

#Utility to get image size withouth loading the image
def get_image_size(path):
    with Image.open(path) as img:
        return img.size  # (width, height)

#Utility to concatenate two images horizontally
def concatenate_images(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def data_explore(img_train_dir, labels_file, img_test_dir='', show_images=False, files_consistency=False): 
    print('Exploring training dataset')
    dataset = TrainDataset(img_train_dir,labels_file)

    #Here I've used the dataframe to manually inspect the image and the class it belongs to. 
    #I've discovered the classifier task is to classify 
    #between persons having or not attributes like hats, glasses, caps, etc
    pos_imgs=dataset.df[dataset.df['cohort']==1]['filename'].reset_index(drop=True)
    neg_imgs=dataset.df[dataset.df['cohort']==0]['filename'].reset_index(drop=True)

    #Just to show some image samples
    if show_images:
        nim=0
        for (im1, im2) in zip(list(pos_imgs), list(neg_imgs)):
            if nim>2:
                break
            dst = concatenate_images(Image.open(os.path.join(img_train_dir, im1 )),
                                     Image.open(os.path.join(img_train_dir, im2 )))
            dst.show()
            nim +=1
    
    
    #Verify files consistency
    existing_files = [f for f in os.listdir(img_train_dir) if '.jpg' in f]
    if files_consistency:
        missing_files = []
        for f in dataset.image_paths:
            if f not in existing_files:
                missing_files.append(f)
        if len(missing_files)>0:
            print('Check  missing files', missing_files)
        else:
            print('Consistency test passed')

    #Get image original resolution
    res = []
    # for f in files:
    #     res.append(get_image_size(os.path.join(img_train_dir, f)))
    # dataset ['res'] = res
    #Also check and create test data
        
    #Then count classes to check if data is well balanced
    counts = dataset.df.groupby('cohort') 
    p_class_0 = 100*len(counts.groups[0])/len(dataset.df)
    p_class_1 = 100*len(counts.groups[1])/len(dataset.df)
    print('There are ',len(dataset.image_paths),' images for training')
    print(counts.count())
    print("Percentages: class 0: ", p_class_0, " % class 1: ", p_class_1, " %")
    if p_class_0<30 or p_class_1<30:
        print('Data is strongly imbalanced')
    
    if img_test_dir!= '':
        print('Exploring testing dataset')
        dataset = TestDataset(img_test_dir, None)
        print('There are ',len(dataset.image_names),' images for testing')
        #Verify image resolution,
        
if __name__ == "__main__":
    labels_file = '/media/alfonso/data2/classifier/ml_exercise_therapanacea/label_train.txt'
    imgs_ds     = '/media/alfonso/data2/classifier/ml_exercise_therapanacea/train_img'
    data_explore(imgs_ds, labels_file,show_images=True, files_consistency=True)