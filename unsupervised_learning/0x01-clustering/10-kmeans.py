#!/usr/bin/env python3
"""Module kmeans."""
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means on a dataset.

    Args:
        X (ndarray): contains the dataset
        k (int): number of clusters
    Returns:
        C, clss
    """
