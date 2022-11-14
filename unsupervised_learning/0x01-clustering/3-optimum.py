#!/usr/bin/env python3
"""Module optimum_k."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Test for the optimum number of clusters by variance.

    Args:
        X (ndarray): contains the data set
        kmin (int, optional): contains the minimum number of clusters to check
                              for. Defaults to 1.
        kmax (int, optional): contains the maximum number of clusters to check
                              for. Defaults to None.
        iterations (int, optional): contains the maximum number of iterations.
                                    Defaults to 1000.
    Returns:
        results, d_vars or None, None
    """
