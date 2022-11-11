#!/usr/bin/env pyhton3
"""Module likelihood."""
import numpy as np


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining this data.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (ndarray): contains the various hypothetical probabilities
    Returns:
        ndarray containing the likelihood of obtaining the data
    """
