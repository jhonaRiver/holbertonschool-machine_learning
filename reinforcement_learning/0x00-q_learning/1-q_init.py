#!/usr/bin/env python3
"""Module q_init."""
import numpy as np


def q_init(env):
    """
    Initialize the Q-table.

    Args:
        env (obj): FrozenLakeEnv instance
    Returns:
        Q-table
    """
    action = env.action_space.n
    states = env.observation_space.n
    q_table = np.zeros((states, action))
    return q_table
