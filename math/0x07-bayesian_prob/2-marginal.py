#!/usr/bin/env python3
"""Module marginal."""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculate the marginal probability of obtaining the data.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (ndarray): contains the various hypothetical probabilities
        Pr (ndarray): contains the prior beliefs about P
    Returns:
        marginal probability
    """
