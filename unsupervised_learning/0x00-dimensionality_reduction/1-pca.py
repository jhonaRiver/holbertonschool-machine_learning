#!/usr/bin/env python3
"""Module pca."""
import numpy as np


def pca(X, ndim):
    """
    Perform PCA on a dataset.

    Args:
        X (ndarray): contains dataset
        ndim (int): new dimensionality of the transformed X
    Returns:
        transformed version of X
    """
    X_mean = X - np.mean(X, axis=0)
    u, Sigma, vh = np.linalg.svd(X_mean)
    W = vh.T
    Wr = W[:, :ndim]
    T = np.dot(X_mean, Wr)
    return T
