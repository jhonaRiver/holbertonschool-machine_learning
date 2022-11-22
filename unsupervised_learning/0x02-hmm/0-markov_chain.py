#!/usr/bin/env python3
"""Module markov_chain."""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determine the probability of a markov chain being in a particular state.

    Args:
        P (ndarray): represents the transition matrix
        s (ndarray): represents the probability of starting in each state
        t (int, optional): number of iterations that the markov chain has been
                           through. Defaults to 1.
    Returns:
        ndarray representing the probability of being in a specific state or
        None
    """
    try:
        if len(P.shape) != 2:
            return None
        n1, n2 = P.shape
        if (n1 != n2) or type(P) is not np.ndarray or not isinstance(t, int):
            return None
        if t < 0:
            return None
        if n1 != s.shape[1] or s.shape[0] != 1:
            return None
        for i in range(t):
            s = np.dot(s, P)
        return s
    except Exception:
        return None
