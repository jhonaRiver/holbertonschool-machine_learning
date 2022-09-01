#!/usr/bin/env python3
"""
module shuffle_data
"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way
    Args:
        X (ndarray): first matrix to shuffle
        Y (ndarray): second matrix to shuffle
    Returns:
        shuffled X and Y matrices
    """
    X = np.random.permutation(X)
    Y = np.random.permutation(Y)
    return X, Y
