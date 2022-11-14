#!/usr/bin/env python3
"""Module agglomerative."""
import numpy as np
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Perform agglomerative clustering on a dataset.

    Args:
        X (ndarray): contains the dataset
        dist (int): maximum cophenetic distance for all clusters
    Returns:
        clss, contains the cluster indices
    """
