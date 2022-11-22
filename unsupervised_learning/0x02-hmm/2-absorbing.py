#!/usr/bin/env python3
"""Module absorbing."""
import numpy as np


def absorbing(P):
    """
    Determine if a markov chain is absorbing.

    Args:
        P (ndarray): represents the standard transition matrix
    Returns:
        True if absorbing or False
    """
    if len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if (n1 != n2) or type(P) is not np.ndarray:
        return None
    D = np.diagonal(P)
    if (D == 1).all():
        return True
    if not (D == 1).any():
        return False
    for i in range(n1):
        for j in range(n2):
            if (i == j) and (i + 1 < len(P)):
                if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                    return False
    return True
