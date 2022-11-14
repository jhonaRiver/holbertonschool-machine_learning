#!/usr/bin/emv python3
"""Module expectation."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X (ndarray): contains the data set
        pi (ndarray): contains the priors
        m (ndarray): contains the centroid means
        S (ndarray): contains the covariance matrices
    Returns:
        g, l or None, none
    """
