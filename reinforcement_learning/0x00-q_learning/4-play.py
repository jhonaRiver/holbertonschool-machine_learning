#!/usr/bin/env python3
"""Module play."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Play an episode.

    Args:
        env (obj): FrozenLakeEnv instance
        Q (ndarray): contains the Q-table
        max_steps (int, optional): maximum number of steps in the episode.
                                   Defaults to 100.
    Returns:
        total rewards for the episode
    """
    state = env.reset()
    env.render()
    done = False
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        if done is True:
            env.render()
            return reward
        env.render()
        state = new_state
    env.close()
    return reward
