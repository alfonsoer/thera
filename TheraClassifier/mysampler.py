#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:23:52 2025

@author: alfonso
"""
import torch
import numpy as np
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
# Function to calculate a sampler, to compesate class imbalance
def get_sampler(trainset, num_clases, batch_size_tr):
    class_sample_count = []
    x = range(0, num_clases) 
    for n in x:
        class_sample_count.append(trainset.labels.count(n)) 
    
    weights_Cls = 1 / torch.Tensor(class_sample_count)
    weights_Cls = weights_Cls.double()
    weights = weights_Cls[trainset.labels]
    
    ns = int(sum(class_sample_count))#Total samples
    
    orig_nbatches = np.ceil(ns/batch_size_tr)
    #Extra_Batches = int(2*orig_nbatches) #We will skip unbalanced batches at training
    #if num_clases <=1:
    #    Extra_Batches = 0
    Extra_Batches = 1
    # Extra_Batches = 50 #We will skip unbalanced batches at training
    nbatches = orig_nbatches + Extra_Batches
    oversampling = int((nbatches)*batch_size_tr)
    num_samples = oversampling
        
    replacement =True
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples, replacement = replacement)
    return sampler