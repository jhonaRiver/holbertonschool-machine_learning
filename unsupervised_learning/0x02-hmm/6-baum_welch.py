#!/usr/bin/env python3
"""Module baum_welch."""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Perform the Bauch-Welch algorithm for a hidden markov model.

    Args:
        Observations (ndarray): contains the index of the observation
        Transition (ndarray): contains the initiliazed transition probabilities
        Emission (ndarray): contains the intiliazed emission probabilities
        Initial (ndarray): contains the initiliazed starting probabilities
        iterations (int, optional): number of times expectation-maximization
                                    should be performed. Defaults to 1000.
    Returns:
        converged Transition, Emission or None, None
    """
