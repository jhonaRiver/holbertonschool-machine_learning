#!/usr/bin/env python3
"""Module regular."""
import numpy as np


def regular(P):
    """
    Determine the steady state probabilities of a regular markov chain.

    Args:
        P (ndarray): represents the transition matrix
    Returns:
        ndarray containing the steady state probabilities or None
    """
