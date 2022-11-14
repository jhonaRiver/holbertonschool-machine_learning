#!/usr/bin/env python3
"""Module variance."""
import numpy as np


def variance(X, C):
    """
    Calculate the total intra-cluster variance.

    Args:
        X (ndarray): contains the data set
        C (ndarray): contains the centroid means for each cluster
    Returns:
        var or None
    """
