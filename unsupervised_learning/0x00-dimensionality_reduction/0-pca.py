#!/usr/bin/env python3
"""Module pca."""
import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on a dataset.

    Args:
        X (ndarray): contains dataset
        var (float, optional): fraction of the variance. Defaults to 0.95.
    Returns:
        weights matrix
    """
    u, Sigma, vh = np.linalg.svd(X, full_matrices=False)
    cumulative_var = np.cumsum(Sigma) / np.sum(Sigma)
    r = (np.argwhere(cumulative_var >= var))[0, 0]
    w = vh.T
    wr = w[:, :r + 1]
    return wr
