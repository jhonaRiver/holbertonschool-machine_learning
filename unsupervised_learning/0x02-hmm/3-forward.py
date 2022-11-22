#!/usr/bin/env python3
"""Module forward."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Perform the forward algorithm for a hidden markov model.

    Args:
        Observation (ndarray): contains the index of the observation
        Emission (ndarray): contains the emission probability of a specific
                            observation given a hidden state
        Transition (ndarray): contains the transition probabilities
        Initial (ndarray): contains the probability of starting in a
                           particular hidden state
    Returns:
        P: likelihood of the observations
        F: contains the forward path probabilities
        None: failure
    """
