#!/usr/bin/env python3
"""
module sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix
    Args:
        confusion (ndarray): confusion matrix where row indices represent the
                             correct labels and column indices represent the
                             predicted labels
    Returns:
        ndarray containing the sensitivity of each class
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    sens = TP / (TP + FN)
    return sens
