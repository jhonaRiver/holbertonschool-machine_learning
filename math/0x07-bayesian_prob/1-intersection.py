#!/usr/bin/env python3
"""Module intersection."""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining this data.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (ndarray): contains the various hypothetical probabilities
        Pr (ndarray): contains the prior beliefsof P
    Returns:
        ndarray containing the intersection
    """
