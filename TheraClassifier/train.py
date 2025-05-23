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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from model import SimpleCNN
from dataset import TrainDataset
from utils import compute_metrics
import os
import copy as cp

def train_model(image_dir, labels_csv, epochs=10, lr=0.001):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(0,translate=(0.015,0.015)),
        transforms.ToTensor()
    ])

    #Check cuda
    num_workers=6
    device_id = 0     
    cuda_list= [ 'cuda:0',
                 'cuda:1',
                 'cuda:2',
                 'cuda:3'
            ]
    
    #Read the dataset
    dataset     = TrainDataset(image_dir, labels_csv, transform=transform, 
                               train_transform=None, stage='')

    #Dataset splitting into K=5 stratified folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    labels = [label for _, label in dataset]

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
        print(f"Fold {fold}")
        #For each fold evaluate on training and validation subsets        
        tr_subset  = cp.copy(dataset)
        val_subset = cp.copy(dataset)
        tr_subset.subset(train_idx) 
        val_subset.subset(val_idx) 
        #Append extra tranformations to training data
        tr_subset.appendTrainTransform(train_transform)
        
        train_loader = DataLoader(tr_subset, batch_size=32, 
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=32,
                                num_workers=num_workers)
        #Model setup
        model = SimpleCNN()
        model = model.cuda() if torch.cuda.is_available() else model
        #Using Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #And a binary cross entropy loss function to minimize
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            print(f"  Epoch {epoch + 1}/{epochs}")
            model.train()
            train_labels = []
            train_probs = []
            for inputs, labels in train_loader:
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                labels = labels.float().unsqueeze(1).cuda() if torch.cuda.is_available() else labels.float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                    train_probs.extend(probs.tolist())
                    train_labels.extend(labels.squeeze().cpu().numpy().tolist())

            train_metrics = compute_metrics(train_labels, train_probs)
            print(f"    Train -> HTER: {train_metrics['HTER']:.4f}, FAR: {train_metrics['FAR']:.4f}, FRR: {train_metrics['FRR']:.4f}, "
                  f"Balanced Acc: {train_metrics['Balanced Accuracy']:.4f}, AUC: {train_metrics['AUC']:.4f}")

            # Validation after epoch
            model.eval()
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                    labels = labels.cuda() if torch.cuda.is_available() else labels
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())

            metrics = compute_metrics(all_labels, all_probs)
            print(f"    Val   -> HTER: {metrics['HTER']:.4f}, FAR: {metrics['FAR']:.4f}, FRR: {metrics['FRR']:.4f}, "
                  f"Balanced Acc: {metrics['Balanced Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")

        torch.save(model.state_dict(),  os.path.join(os.path.dirname(image_dir),'models','fold_'+str(fold)+'.pt'))
        print(f"Model saved for fold {fold}")

if __name__ == "__main__":
    labels_file = '/media/alfonso/data2/classifier/ml_exercise_therapanacea/label_train.txt'
    imgs_ds     = '/media/alfonso/data2/classifier/ml_exercise_therapanacea/train_img'
    train_model(imgs_ds, labels_file, epochs=12,lr=0.001)