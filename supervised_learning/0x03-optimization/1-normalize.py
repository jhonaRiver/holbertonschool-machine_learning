#!/usr/bin/env python3
"""
module normalize
"""


def normalize(X, m, s):
    """
    normalizes a matrix
    Args:
        X (ndarray): matrix to normalize
        m (ndarray): contains the mean of all features of X
        s (ndarray): contains the standard deviation of all features of X
    Returns:
        normalized X matrix
    """
    return (X - m) / s
