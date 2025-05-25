#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:20:49 2025

@author: alfonso
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from model import SimpleCNN, SimpleCNNBN,SimpleCNNBNDr, ResNetBinary
from dataset import TrainDataset
from utils import compute_metrics, plot_metrics
import os
import copy as cp
import pandas as pd
import torch.nn.functional as F

def train_model(image_dir, labels_csv, epochs=10, lr=0.001,
                step_size=5, gamma=0.7, save_dir='', cnn_type='baseline',
               SEED= 10000, scheduler_str='StepLR', input_size=(64,64),     
               batch_size =32):
    if cnn_type=='resnet':
        input_size = (254,254)
    transform = transforms.Compose([
        #transforms.Resize(input_size), #To improve resizing at loading time, I will interpolate the entire batch in tensor form on gpu.
        transforms.ToTensor()
    ])

    train_transform = transforms.Compose([
        #transforms.Resize(input_size),
        transforms.RandomAffine(0,translate=(0.05,0.05)),
        #Images present always a portrait of the person,   
        #there is no need to rotate more than -5 to 5 degrees 
        transforms.RandomRotation(degrees=(-5, 5)),
        #transforms.Resize(input_size),  #To improve resizing at loading time, I will interpolate the entire batch in tensor form on gpu. #In some cases it may be unsafe to do this because of different image sizes
        transforms.ToTensor()
    ])
    
    #Create output dir, log_dir and model_dir
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, 'logs')
    model_dir = os.path.join(save_dir, 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    #Check GPU disponibility
    move_to_gpu = False
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        move_to_gpu=True  
        torch.cuda.empty_cache()  
        torch.cuda.manual_seed(SEED) 
        
                    
    #Useful when having mutiple GPUs
    device_id = 0     
    cuda_list= [ 'cuda:0',
                 'cuda:1',
                 'cuda:2',
                 'cuda:3'
            ]
    #Number of threads for the dataloader to read data simultaneously 
    num_workers=8
    #Read the dataset
    dataset     = TrainDataset(image_dir, labels_csv, transform=transform, 
                               train_transform=None, stage='')
    labels = [label for _, label in dataset]

    #Dataset splitting into K=5 stratified folds,
    #assuring samples of the minority class are taken into account
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
        print(f"Fold {fold}")
        #For each fold evaluate on training and validation subsets        
        tr_subset  = cp.copy(dataset)
        val_subset = cp.copy(dataset)
        tr_subset.subset(train_idx) 
        val_subset.subset(val_idx)
        
        #Modify tranformations to training data
        tr_subset.modifyTransform(train_transform)

        #Dataloaders for reading training and validation samples
        train_loader = DataLoader(tr_subset, batch_size=batch_size, 
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size,
                                num_workers=num_workers)
        #Model setup. 
        model = None
        if cnn_type=='baseline':
            model = SimpleCNN()
        elif cnn_type=='BNbaseline':
            model = SimpleCNNBN()
        elif cnn_type=='BNbaselineDr':
            model = SimpleCNNBNDr()
        elif cnn_type=='resnet':
            model = ResNetBinary()

        if model is None:
            raise Exception("Can't continue due to undefined training model.")            
 
        #Verify GPU availability. This is also done for the inputs as well as for the labels
        model = model.cuda() if move_to_gpu else model
        #Using Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #And a binary cross entropy loss function to minimize
        criterion = nn.BCEWithLogitsLoss()
        #I will set an scheduler to control the learning rate
        scheduler = None
        if scheduler_str == 'StepLR':
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) 
        
        history = {'train_metrics': [],  'val_metrics': []}
        for epoch in range(epochs):
            lr_start = optimizer.state_dict()['param_groups'][0]['lr']
            print(f"  Epoch {epoch + 1}/{epochs}, lr={lr_start:.8f}")
            #Model mode as training
            model.train()
            train_labels = []
            train_probs = []
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.cuda() if move_to_gpu else inputs
                #Interpolate the entire batch
                inputs = F.interpolate(inputs, input_size)
                labels = labels.float().unsqueeze(1).cuda() if move_to_gpu else labels.float().unsqueeze(1)
                optimizer.zero_grad() #Gradients init before each evaluation
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward() #Loss backpropagation
                optimizer.step()
                epoch_loss+= loss.item() * inputs.size(0)
                
                with torch.no_grad():#Reduce memory use by disabling gradient calculation
                    #.cpu transfers data from GPU to the CPU if is the case
                    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                    train_probs.extend(probs.tolist())
                    train_labels.extend(labels.squeeze().cpu().numpy().tolist())
                    
            epoch_train_loss = epoch_loss / len(train_loader.dataset)
            train_metrics = compute_metrics(train_labels, train_probs)
            train_metrics['loss'] =  epoch_train_loss
            history['train_metrics'].append(train_metrics) 
            
            print(f"    Train -> Loss: {train_metrics['loss']:.4f}, HTER: {train_metrics['HTER']:.4f}, FAR: {train_metrics['FAR']:.4f}, FRR: {train_metrics['FRR']:.4f}, "
                  f"Balanced Acc: {train_metrics['Balanced Accuracy']:.4f}, AUC: {train_metrics['AUC']:.4f}")

            # Validation after epoch
            model.eval()
            all_labels = []
            all_probs = []
            epoch_loss = 0.0
            with torch.no_grad(): #Reduce memory use by disabling gradient calculation
                for inputs, labels in val_loader:
                    inputs = inputs.cuda() if move_to_gpu else inputs
                    #Interpolate the entire batch
                    inputs = F.interpolate(inputs, input_size)
                    labels = labels.cuda() if move_to_gpu else labels
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    epoch_loss+= loss.item() * inputs.size(0)
            epoch_val_loss = epoch_loss / len(val_loader.dataset)
            val_metrics = compute_metrics(all_labels, all_probs)
            val_metrics['loss'] =  epoch_val_loss
            history['val_metrics'].append(val_metrics)
            scheduler.step() #Step to next learning rate change
            print(f"    Val   -> Loss: {val_metrics['loss']:.4f}, HTER: {val_metrics['HTER']:.4f}, FAR: {val_metrics['FAR']:.4f}, FRR: {val_metrics['FRR']:.4f}, "
                  f"Balanced Acc: {val_metrics['Balanced Accuracy']:.4f}, AUC: {val_metrics['AUC']:.4f}")
            
        torch.save(model.state_dict(),  os.path.join(model_dir,'fold_'+str(fold)+'.pt'))
        print(f"Model saved for fold {fold}")

        # Save metrics and loss history
        train_metrics_df = pd.DataFrame(history['train_metrics'])
        val_metrics_df = pd.DataFrame(history['val_metrics'])
        train_metrics_df.to_csv(os.path.join(log_dir, f'train_metrics_fold_{fold}.csv'), index=False)
        val_metrics_df.to_csv(os.path.join(log_dir, f'val_metrics_fold_{fold}.csv'), index=False)
        # Plot HTER and loss 
        plot_metrics(train_metrics_df, val_metrics_df, fold, log_dir, metric='HTER')
        plot_metrics(train_metrics_df, val_metrics_df, fold, log_dir, metric='loss')
        plot_metrics(train_metrics_df, val_metrics_df, fold, log_dir, metric='Balanced Accuracy')

        #Free GPU memory
        if move_to_gpu:
            torch.cuda.empty_cache()    

        
if __name__ == "__main__":
    labels_file = '/home/sagemaker-user/data/label_train.txt'
    imgs_ds     = '/home/sagemaker-user/data/train_img'
    save_dir    = '/home/sagemaker-user'
    train_model(imgs_ds, labels_file, epochs=5,lr=0.001,save_dir=save_dir)
