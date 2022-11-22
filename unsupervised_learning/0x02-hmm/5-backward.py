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
    try:
        T = Observation.shape[0]
        N, M = Emission.shape
        beta = np.zeros((N, T))
        beta[:, T - 1] = np.ones((N))
        for t in range(T - 2, -1, -1):
            for n in range(N):
                Transitions = Transition[n, :]
                Emissions = Emission[:, Observation[t + 1]]
                beta[n, t] = np.sum((Transitions * beta[:, t + 1]) * Emissions)
        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])
        return P, beta
    except Exception:
        return None, None
