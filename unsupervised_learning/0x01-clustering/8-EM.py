#!/usr/bin/env python3
"""Module expectation_maximization."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Perform the expectation maximization for a GMM.

    Args:
        X (ndarray): contains the data set
        k (int): contains the number of clusters
        iterations (int, optional): contains the maximum number of iterations.
                                    Defaults to 1000.
        tol (float, optional): contains tolerance of the log likelihood, used
                               to determine early stopping. Defaults to 1e-5.
        verbose (bool, optional): determines if you should print information.
                                  Defaults to False.
    Returns:
        pi, m, S, g, l or None, None, None, None, None
    """
