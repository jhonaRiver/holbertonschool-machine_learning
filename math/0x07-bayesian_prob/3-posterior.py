#!/usr/bin/env python3
"""Module posterior."""
import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculate the posterior probability for the various hypothetical
    probabilities.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (ndarray): contains the various hypothetical probabilities
        Pr (ndarray): contains the prior beliefs of P
    Returns:
        posterior probability
    """
