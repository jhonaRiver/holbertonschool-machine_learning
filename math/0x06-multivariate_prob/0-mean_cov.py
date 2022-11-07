#!/usr/bin/env python3
"""Module mean-cov."""
import numpy as np


def mean_cov(X):
    """
    Calculate the mean and covariance of a data set.

    Args:
        X (ndarray): contains the data set
    Returns:
        mean, cov
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n, d = X.shape
    mean = np.mean(X, axis=0).reshape(1, d)
    X_mean = X - mean
    cov = np.matmul(X_mean.T, X_mean) / (n - 1)
    return mean, cov
