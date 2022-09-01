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
    m = X.shape[0]
    means = []
    std_devs = []
    for i in range(X.shape[1]):
        mean = (1 / m) * np.sum(X[i])
        std_dev = (1 / m) * (np.sum((X[i] - mean) ** 2))
        means.append(mean)
        std_devs.append(std_dev)
    return means, std_devs
