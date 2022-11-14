#!/usr/bin/env python3
"""Module initialize."""
import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroids for K-means.

    Args:
        X (ndarray): contains the dataset that will be used
        k (int): contains the number of clusters
    Returns:
        ndarray containing the initialized centroids for each cluster or None
    """
