#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:29:32 2025

@author: alfonso
"""

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import balanced_accuracy_score
import numpy as np

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