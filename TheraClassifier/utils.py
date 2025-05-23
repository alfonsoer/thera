#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:29:32 2025

@author: alfonso
"""

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (np.array(y_pred_probs) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn + 1e-6)
    frr = fn / (fn + tp + 1e-6)
    hter = 0.5 * (far + frr)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)
    return {
        'HTER': hter,
        'FAR': far,
        'FRR': frr,
        'Balanced Accuracy': bal_acc,
        'AUC': auc
    }

def plot_metrics(train_metrics_df, val_metrics_df,  fold, log_dir, metric='loss'):
        plt.figure()
        plt.plot(train_metrics_df[metric].tolist(), label='Train '+metric)
        plt.plot(val_metrics_df[metric].tolist(), label='Validation '+metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} for fold {fold}')
        plt.legend()
        plt.savefig(os.path.join(log_dir, f'{metric}_fold_{fold}.pdf'))
        plt.close()
        
 