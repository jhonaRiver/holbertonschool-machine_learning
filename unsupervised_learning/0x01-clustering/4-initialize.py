#!/usr/bin/env python3
"""Module initialize."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize variables for a Gaussian Mixture Model.

    Args:
        X (ndarray): contains the data set
        k (int): contains the number of clusters
    Returns:
        pi, m, S or None, None, None
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None
    n, d = X.shape
    phi = np.ones(k)/k
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)
    return phi, m, S
