#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:03:16 2025

@author: alfonso
"""

import os
import pandas as pd
from PIL import Image
import imagesize

#Utility to get imaga size withouth loading the image
def get_image_size(path):
    with Image.open(path) as img:
        return img.size  # (width, height)

#Utility to concatenate two images horizontally
def concatenate_images(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def data_explore(images_dir, labels_file, show_images=False):    
    # labels_file = '/Users/alfonso/Documents/Chamba/Therapanacea/Excercise/ml_exercise_therapanacea/label_train.txt'
    # imgs_ds     = '/Users/alfonso/Documents/Chamba/Therapanacea/Excercise/ml_exercise_therapanacea/train_img/'
    #Read labels file
    
    dataset          = pd.read_csv(labels_file,header=None, names=['cohort'])
    #Create a pd column with image files. File names are formatted accordingly
    files = []
    K='0'
    for r in range(1,dataset.cohort.size+1): 
        file = str(r)
        N = len(str(dataset.cohort.size))-len(file)
        file=K*N+file+'.jpg'
        files.append(file)
    dataset ['filename'] = files
    #Here I've used the dataframe to manually inspect the image and the class it belongs to. 
    #I discovered the classifier task is to classify between persons having or not attributes like
    #hats, glasses, cap, etc
    pos_imgs=dataset[dataset['cohort']==1]['filename'].reset_index(drop=True)
    neg_imgs=dataset[dataset['cohort']==0]['filename'].reset_index(drop=True)
    
    if show_images:
        nim=0
        for (im1, im2) in zip(list(pos_imgs), list(neg_imgs)):
            if nim>2:
                break
            dst = concatenate_images(Image.open(os.path.join(images_dir, im1 )),
                                     Image.open(os.path.join(images_dir, im2 )))
            dst.show()
            nim +=1
    
    
    #Verify files consistency
    existing_files = [f for f in os.listdir(images_dir) if '.jpg' in f]
    # missing_files = []
    # for f in files:
    #     if f not in existing_files:
    #         missing_files.append(f)
    # if len(missing_files)>0:
    #     print('Check  missing files', missing_files)
    # else:
    #     print('Consistency test passed')

    #Get image original resolution
    res = []
    # for f in files:
    #     res.append(get_image_size(os.path.join(images_dir, f)))
    # dataset ['res'] = res
    #Also check and create test data
        
    
    #Then count classes to check if data is well balanced
    counts = dataset.groupby('cohort') 
    p_class_0 = 100*len(counts.groups[0])/len(dataset)
    p_class_1 = 100*len(counts.groups[1])/len(dataset)
    
    print(counts.count())
    print("Percentages: class 0: ", p_class_0, " % class 1: ", p_class_1, " %")
    if p_class_0<30 or p_class_1<30:
        print('Data is strongly imbalanced')
    

if __name__ == "__main__":
    labels_file = '/Users/alfonso/Documents/Chamba/Therapanacea/Excercise/ml_exercise_therapanacea/label_train.txt'
    imgs_ds     = '/Users/alfonso/Documents/Chamba/Therapanacea/Excercise/ml_exercise_therapanacea/train_img/'
    data_explore(imgs_ds, labels_file)