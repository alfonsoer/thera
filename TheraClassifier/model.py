#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:25:28 2025

@author: alfonso
"""

import torch.nn as nn
from torchvision import models
        
class ResNetBinary(nn.Module):
    def __init__(self, pretrained=False, num_classes=1): #classes=1 since I am using BCEWithLogitsLoss
        super(ResNetBinary, self).__init__()
        # Load pretrained ResNet 
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace final fc layer with  num_classes
        self.model.fc = nn.Sequential(
                            nn.Dropout(0.25),
                            nn.Linear(self.model.fc.in_features, num_classes),
                            #Add extra linear layers
                            #nn.BatchNorm1d(128),
                            #nn.ReLU(),
                            #nn.Dropout(0.2),
                            #nn.Linear(128, num_classes)
                            )
        
    def forward(self, x):
        return self.model(x)