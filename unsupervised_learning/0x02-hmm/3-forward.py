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
    try:
        N = Transition.shape[0]
        T = Observation.shape[0]
        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]
        for t in range(1, T):
            for n in range(N):
                Transitions = Transition[:, n]
                Emissions = Emission[n, Observation[t]]
                F[n, t] = np.sum(Transitions * F[:, t - 1] * Emissions)
        P = np.sum(F[:, -1])
        return P, F
    except Exception:
        None, None
