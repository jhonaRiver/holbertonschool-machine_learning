#!/usr/bin/env python3
"""Module posterior."""
import numpy as np
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculate the posterior probability that the probability of developing\
        severe side effects falls within a specific range given the data.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        p1 (float): lower bound on the range
        p2 (float): upper bound on the range
    Returns:
        posterior probability that p is within the range
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal\
            to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or not 0 <= p1 <= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if type(p2) is not float or not 0 <= p2 <= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    def likelihood(x, n, P):
        return (special.factorial(n) / (special.factorial(x) *
                special.factorial(n - x))) * (P ** x) * ((1 - P) ** (n - x))

    P = (x - (p1 + p2)) / (n - (p1 + p2))
    like = likelihood(x, n, P)
    Pr = special.beta(1, 1)
    intersection = like * Pr
    marginal = np.sum(intersection)
    pos = intersection / marginal
    return pos
