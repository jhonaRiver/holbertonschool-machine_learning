#!/usr/bin/env python3
"""Module BIC."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using the Bayesian Information Criterion.

    Args:
        X (ndarray): contains the data set
        kmin (int, optional): contains the minimum number of clusters. Defaults to 1.
        kmax (int, optional): contains the maximum number of clusters. Defaults to None.
        iterations (int, optional): contains the maximum number of iterations. Defaults to 1000.
        tol (float, optional): contains the tolerance. Defaults to 1e-5.
        verbose (bool, optional): determines if the EM algorithm should print information. Defaults to False.
    Returns:
        best_k, best_result, l, b or None, None, None, None
    """
