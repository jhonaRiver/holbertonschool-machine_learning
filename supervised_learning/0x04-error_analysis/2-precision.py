#!/usr/bin/env python3
"""
module precision
"""
import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix
    Args:
        confusion (ndarray): confusion matrix where row indices represent the
                             correct labels and column indices represent the
                             predicted labels
    Returns:
        ndarray containing the precision of each class
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    prec = TP / (TP + FP)
    return prec
