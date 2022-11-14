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
