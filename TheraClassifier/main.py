#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:17:11 2025

@author: alfonso
"""

import argparse
from data_exploration import data_explore
from train import train_model
from validate import validate_model
from test import test_model

def main():
    #Parser arguments to run main from command line
    parser = argparse.ArgumentParser(description="Binary image classifier")
    parser.add_argument('--mode', choices=['explore','train', 'test'], required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--labels_txt', type=str, required=False)
    parser.add_argument('--model_dir', type=str, required=False)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--lr', type=str, required=False)
    parser.add_argument('--step', type=int, required=False)
    parser.add_argument('--gamma', type=str, required=False)

    #Default training parameters
    epochs = 12
    lr=0.001
    step_size=5
    gamma=0.7

    args = parser.parse_args()
    
    if args.mode == 'explore':
        if not args.labels_txt:
            raise ValueError("Required --labels_txt for data exploration.")
        data_explore(args.image_dir, args.labels_txt)
    elif args.mode == 'train':
        if not args.labels_txt:
            raise ValueError("Required --labels_txt for training.")  

        if args.epochs:
            epochs=int( args.epochs)
        if args.lr:
            lr=float( args.lr)   
        if args.step:
            step_size=int( args.step)
        if args.gamma:
            gamma=float( args.gamma)   
        
        train_model(args.image_dir, args.labels_txt, 
                    epochs=epochs, lr=lr,step_size=step_size,gamma=gamma  )
    elif args.mode == 'test':
        if not args.model_dir:
            raise ValueError("Required --model_dir for test.")
        test_model(args.image_dir, args.model_dir)

if __name__ == '__main__':
    main()
