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
from dataset import BinaryImageDataset
from utils import compute_metrics
import os


def train_model(image_dir, labels_csv):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = BinaryImageDataset(image_dir, labels_csv, transform)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    labels = [label for _, label in dataset]

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
        print(f"Fold {fold}")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=32)

        model = SimpleCNN()
        model = model.cuda() if torch.cuda.is_available() else model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(10):
            model.train()
            for inputs, labels in train_loader:
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                labels = labels.float().unsqueeze(1).cuda() if torch.cuda.is_available() else labels.float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), f'models/fold_{fold}.pt')
        print(f"Model saved for fold {fold}")

if __name__ == "__main__":
    labels_file = '/Users/alfonso/Documents/Chamba/Therapanacea/Excercise/ml_exercise_therapanacea/label_train.txt'
    imgs_ds     = '/Users/alfonso/Documents/Chamba/Therapanacea/Excercise/ml_exercise_therapanacea/train_img/'
    train_model(imgs_ds, labels_file)