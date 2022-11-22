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
