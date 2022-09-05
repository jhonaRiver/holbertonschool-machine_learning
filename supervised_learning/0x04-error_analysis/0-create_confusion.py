#!/usr/bin/env python3
"""
module create_confusion_matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    Args:
        labels (ndarray): contains the correct labels for each data point
        logits (ndarray): contains the predicted labels
    Returns:
        confusion matrix with row indices representing the correct labels and
        column indices representing the predicted labels
    """
    return np.matmul(labels.T, logits)
