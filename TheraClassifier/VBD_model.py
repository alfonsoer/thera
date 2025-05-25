#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:25:28 2025

@author: alfonso
Simple CNN based on my paper: https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.26009
https://github.com/alfonsoer/vbd.git
"""

import torch
import torch.nn as nn
from torch.nn import ModuleList
import numpy as np

def computeSize(N, kernel_size, stride, padding):
    """
    Precomputes the output size of a convolutional or maxpooling layer
    """    
    return int(np.floor(((N + 2*padding)-(kernel_size-stride)) / stride))

class vbdSimpleCnn(nn.Module):
    
    default_conv_layer_widths = [[(16,3)],[(32,3)], [(64,3)]] 
    default_linear_layer_widths = [64,128, 1]  
    
    def __init__(self, image_size, input_channels, conv_layer_widths, linear_layer_widths, prob=0.5, use_bn=False):
        super(vbdSimpleCnn,self).__init__()
        self.image_size = image_size
        eff_image_size = [i for i in image_size]
        
        self.downsample = nn.MaxPool2d(kernel_size=2,stride=2)

        MP_kernel_size = 3
        MP_stride = 2
        MP_padding=1
        self.downsample = nn.MaxPool2d(kernel_size=MP_kernel_size,
                                       stride=MP_stride, 
                                       padding=MP_padding)
        
        #make the convolution layers
        #each is composed of a number of convolutions followed by optional batch norms
        self.res_layers = []
        first_outer=True
        for l_list in conv_layer_widths:
            conv_layers = []
            for layer_width, kernel in l_list:
                conv_layers.append(nn.Dropout2d(prob))
                conv_layers.append(nn.Conv2d(input_channels,layer_width,kernel,padding=int(kernel/2),padding_mode='replicate'))
                torch.nn.init.constant_(conv_layers[-1].bias, 0)
                input_channels = layer_width
                if use_bn:
                    conv_layers.append(nn.BatchNorm2d(input_channels))
                conv_layers.append(nn.ReLU())

            if first_outer:
                conv_layers = conv_layers[1:]
            self.res_layers.append(ModuleList(conv_layers))
            if not first_outer:

                eff_image_size = [computeSize( e,
                                            MP_kernel_size,
                                            MP_stride,
                                            MP_padding) for e in eff_image_size]
            first_outer = False
        self.res_layers = ModuleList(self.res_layers)
        
        #make the linear layers
        input_vector = input_channels*eff_image_size[0]*eff_image_size[1]
        self.lin_layers = []
        for l,layer_width in enumerate(linear_layer_widths):
            if l != len(linear_layer_widths)-1:
                self.lin_layers.append(nn.Dropout(prob))
            self.lin_layers.append(nn.Linear(input_vector,layer_width))
            torch.nn.init.constant_(self.lin_layers[-1].bias, 0)
            if l != len(linear_layer_widths)-1:
                self.lin_layers.append(nn.ReLU())
            input_vector = layer_width
        self.lin_layers = ModuleList(self.lin_layers)
        
        
    def forward(self, img):
        #apply convolution layers
        first = True
        for res in self.res_layers:
            if not first:
                img = self.downsample(img)
            first = False
            for layer in res:
                img = layer(img)

        #apply linear layers
        lin = img.view(img.shape[0],-1)
        for layer in self.lin_layers:
            lin = layer(lin)
        
        return lin