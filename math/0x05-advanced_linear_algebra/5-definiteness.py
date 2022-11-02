#!/usr/bin/env python3
"""Module definiteness."""
import numpy as np


def definiteness(matrix):
    """
    Calculate the definiteness of a matrix.

    Args:
        matrix (ndarray): whose definiteness should be calculated
    Returns:
        positive definite, positive semi-definite, negative semi-definite,
        negative definite, indefinite or None
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    if (matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.all(matrix.T == matrix):
        return None
    w, v = np.linalg.eig(matrix)
    if np.all(w > 0):
        return "Positive definite"
    elif np.all(w >= 0):
        return "Positive semi-definite"
    elif np.all(w < 0):
        return "Negative definite"
    elif np.all(w <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
