#!/usr/bin/env python3
"""
module specificity
"""
import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix
    Args:
        confusion (ndarray): confusion matrix where row indices represent the
                             correct labels and column indices represent the
                             predicted labels
    Returns:
        ndarray containing the specificity of each class
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    spec = TN / (TN + FP)
    return spec
