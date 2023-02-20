#!/usr/bin/env python3
"""Module sarsa_lambtha."""


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform SARSA.

    Args:
        env (obj): openAI environment instance
        Q (ndarray): cotains the Q table
        lambtha (float): eligibility trace factor
        episodes (int, optional): total number of episodes to train over.
        Defaults to 5000.
        max_steps (int, optional): maximum number of steps per episode.
        Defaults to 100.
        alpha (float, optional): learning rate. Defaults to 0.1.
        gamma (float, optional): discount rate. Defaults to 0.99.
        epsilon (int, optional): initial treshold for epsilon greedy.
        Defaults to 1.
        min_epsilon (float, optional): minimum value that epsilon should decay
        to. Defaults to 0.1.
        epsilon_decay (float, optional): decay rate for updating epsilon
        between episodes. Defaults to 0.05.
    Returns:
        Q: updated Q table
    """
