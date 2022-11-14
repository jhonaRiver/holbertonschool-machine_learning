#!/usr/bin/env python3
"""Module gmm."""
import numpy as np
import sklearn.mixture


def gmm(X, k):
    """
    Calculate a GMM from a dataset.

    Args:
        X (ndarray): contains the dataset
        k (int): number of clusters
    Returns:
        pi, m, S, clss, bic
    """
