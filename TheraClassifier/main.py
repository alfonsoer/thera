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
    parser.add_argument('--mode', choices=['explore','train', 'validate', 'test'], required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--labels_txt', type=str, required=False)
    parser.add_argument('--model_dir', type=str, required=False)
    args = parser.parse_args()
    
    if args.mode == 'explore':
        data_explore(args.image_dir, args.labels_txt)
    elif args.mode == 'train':
        train_model(args.image_dir, args.labels_txt)
    # elif args.mode == 'validate':
    #     if not args.model_dir:
    #         raise ValueError("Required --model_dir for validation.")
    #     validate_model(args.image_dir, args.labels_txt, args.model_dir)
    # elif args.mode == 'test':
    #     if not args.model_dir:
    #         raise ValueError("Required --model_dir for test.")
    #     test_model(args.image_dir, args.model_dir)

if __name__ == '__main__':
    main()