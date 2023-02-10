#!/usr/bin/env python3
"""Module epsilon_greedy."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Use epsilon-greedy to determine the next action.

    Args:
        Q (ndarray): contains the q-table
        state (int): current state
        epsilon (float): epsilon to use for calculation
    Returns:
        next action index
    """
    e_tradeoff = np.random.uniform(0, 1)
    if e_tradeoff < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action
