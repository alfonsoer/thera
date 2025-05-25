#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:28:51 2025

@author: alfonso
"""

import torch
import os
import numpy as np
from model import  ResNetBinary
from VBD_model import vbdSimpleCnn
from dataset import get_test_dataloader
from torchvision import transforms
from collections import Counter
import pandas as pd
import torch.nn.functional as F

#Calculate the mojority vote from all classifiers
def majority_vote(predictions_list):
    return [Counter(col).most_common(1)[0][0] for col in zip(*predictions_list)]

#Evaluate each individual classifier calculated for each n-fold
def test_model(image_dir, model_dir, Nfolds=5, cnn_type='baseline', input_size=(64,64),
              input_channels=3, batch_size =32, num_clases=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers=8

    if cnn_type=='resnet':
        input_size = (254,254)
        
    transform = transforms.Compose([
        #transforms.Resize(input_size), #To improve resizing at loading time, I will interpolate the entire batch in tensor form on gpu. #In some cases it may be unsafe to do this because of different image sizes
        transforms.ToTensor()
    ])

    test_loader, image_names = get_test_dataloader(image_dir, transform, 
                                                   batch_size=batch_size,num_workers=8)
    df = pd.DataFrame({'filename': image_names})
    fold_preds = []
    for i in range(Nfolds):
        model_path = os.path.join(model_dir, f'fold_{i}.pt')
        #Create an model instance and then load trained weights
        model = None
        if cnn_type=='resnet':
            model = ResNetBinary().to(device)
        elif cnn_type=='vbdSimpleCnn':
            conv_layer_widths = [[(16,3)],[(32,3),(32,3)], [(64,3),(64,3)], [(128,3),(128,3)]] 
            linear_layer_widths = [64,128, 1] 
            model = vbdSimpleCnn(torch.Size(input_size),input_channels,
                                 conv_layer_widths,linear_layer_widths,
                                 prob=0.25,use_bn=True).to(device)
        if model is None:
            raise Exception("Can't continue due to undefined training model.")   

        #Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        preds = []
        with torch.no_grad():#Reduce memory use by disabling gradient calculation
            #Calculate predictions
            for inputs in test_loader:
                inputs = inputs.to(device)
                inputs = F.interpolate(inputs, input_size)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds.extend((probs >= 0.5).astype(int))
        fold_preds.append(preds)
        df[f'pred_{i+1}']=preds
    final_preds = majority_vote(fold_preds)
    df['prediction'] = final_preds
    
    df.to_csv(os.path.join(os.path.dirname(image_dir),"test_predictions_majority_voting.csv"), index=False)
    df['prediction'].to_csv(os.path.join(os.path.dirname(image_dir),"label_val.txt"), header=None, index=None)