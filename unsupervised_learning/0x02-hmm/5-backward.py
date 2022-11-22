#!/usr/bin/env python3
"""Module backward."""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Perform the backward algorithm for a hidden markov model.

    Args:
        Observation (ndarray): contains the index of the observation
        Emission (ndarray): contains the emission probability of a specific
                            observation given a hidden state
        Transition (ndarray): contains the transition probabilities
        Initial (ndarray): contains the probability of starting in a
                           particular hidden state
    Returns:
        P: likelihood of the observations given the model
        B: contains the backward path probabilities
        None: failure
    """
