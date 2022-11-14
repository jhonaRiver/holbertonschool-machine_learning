#!/usr/bin/env python3
"""Module kmeans."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform K-means on a dataset.

    Args:
        X (ndarray): contains the dataset
        k (int): contains the number of clusters
        iterations (int, optional): contains the maximum number of iterations.
                                    Defaults to 1000.
    Returns:
        C, clss or None, None
    """
