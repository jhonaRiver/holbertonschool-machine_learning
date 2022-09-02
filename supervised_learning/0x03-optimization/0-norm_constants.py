#!/usr/bin/env python3
"""
module normalization_constants
"""
import numpy as np


def normalization_constants(X):
    """
    calculates the normalization constants of a matrix
    Args:
        X (ndarray): matrix to normalize
    Returns:
        mean and standard deviation of each feature
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
