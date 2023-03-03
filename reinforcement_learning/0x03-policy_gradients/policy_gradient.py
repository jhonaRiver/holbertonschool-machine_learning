#!/usr/bin/env python3
"""Module policy."""
import numpy as np


def policy(matrix, weight):
    """
    Compute to policy with a weight of a matrix.

    Args:
        matrix (matrix): matrix to use
        weight (matrix): weight of the matrix
    """
    dot_product = matrix.dot(weight)
    exp = np.exp(dot_product)
    policy = exp / np.sum(exp)
    return policy


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient.

    Args:
        state (matrix): represents the current observation of the environment
        weight (matrix): random weight
    Returns:
        action
        gradient
    """
    Policy = policy(state, weight)
    action = np.random.choice(len(Policy[0]), p=Policy[0])
    s = Policy.reshape(-1, 1)
    softmax = (np.diagflat(s) - np.dot(s, s.T))[action, :]
    dlog = softmax / Policy[0, action]
    gradient = state.T.dot(dlog[None, :])
    return action, gradient
