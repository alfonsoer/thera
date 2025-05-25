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
    parser.add_argument('--img_train_dir', type=str, required=False)
    parser.add_argument('--img_test_dir', type=str, required=False)
    parser.add_argument('--labels_txt', type=str, required=False)
    parser.add_argument('--model_dir', type=str, required=False)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--lr', type=str, required=False)
    parser.add_argument('--step', type=int, required=False)
    parser.add_argument('--gamma', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    args = parser.parse_args()
    
    #Default training parameters
    epochs = 12
    lr=0.001
    step_size=5
    gamma=0.7
    batch_size = 64
    save_dir = ''
    
    if args.mode == 'explore':
        if not args.img_train_dir:
            raise ValueError("Required --img_train_dir for data exploration.")            
        if not args.labels_txt:
            raise ValueError("Required --labels_txt for data exploration.")
        img_test_dir = ''
        if args.img_test_dir:
            img_test_dir= args.img_test_dir
        data_explore(args.img_train_dir, args.labels_txt, img_test_dir=img_test_dir, show_images=True)
    elif args.mode == 'train':        
        if not args.img_train_dir:
            raise ValueError("Required --img_train_dir for training.") 
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
        if args.save_dir:
            save_dir=args.save_dir
        train_model(args.img_train_dir, args.labels_txt, 
                    epochs=epochs, lr=lr,step_size=step_size,gamma=gamma,
                    save_dir=save_dir, cnn_type='resnet', batch_size=batch_size)
    elif args.mode == 'test':
        if not args.model_dir:
            raise ValueError("Required --model_dir for testing.")
        if not args.img_test_dir:
            raise ValueError("Required --img_test_dir for testing.")
        test_model(args.img_test_dir, args.model_dir, cnn_type='resnet')

if __name__ == '__main__':
    main()
