#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:28:51 2025

@author: alfonso
"""

import torch
import os
import numpy as np
from model import SimpleCNN
from dataset import get_test_dataloader
from torchvision import transforms
from collections import Counter
import pandas as pd

def majority_vote(predictions_list):
    return [Counter(col).most_common(1)[0][0] for col in zip(*predictions_list)]

def test_model(image_dir, model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    test_loader, image_names = get_test_dataloader(image_dir, transform)

    fold_preds = []
    for i in range(5):
        model_path = os.path.join(model_dir, f'fold_{i}.pt')
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds.extend((probs >= 0.5).astype(int))
        fold_preds.append(preds)

    final_preds = majority_vote(fold_preds)

    df = pd.DataFrame({'filename': image_names, 'prediction': final_preds})
    df.to_csv("test_predictions_majority_voting.csv", index=False)
    print("✅ Predicciones guardadas en test_predictions_majority_voting.csv")
