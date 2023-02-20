#!/usr/bin/env python3
"""Module td_lambtha."""


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Perform the TD algorithm.

    Args:
        env (obj): openAI environment instance
        V (ndarray): contains the value estimate
        policy (func): takes in a state and returns the next action to take
        lambtha (float): eligibility trace factor
        episodes (int, optional): total number of episodes to train over.
        Defaults to 5000.
        max_steps (int, optional): maximum number of steps per episode.
        Defaults to 100.
        alpha (float, optional): learning rate. Defaults to 0.1.
        gamma (float, optional): discount rate. Defaults to 0.99.
    Returns:
        V: updated value estimate
    """
