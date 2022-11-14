#!/usr/bin/env python3
"""Module maximization."""
import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in the EM algorithm for a GMM.

    Args:
        X (ndarray): contains the data set
        g (ndarray): contains the posterior probabilities
    Returns:
        pi, m, S or None, None, None
    """
