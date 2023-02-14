#!/usr/bin/env python3
"""Module train."""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform Q-learning.

    Args:
        env (obj): FrozenLakeEnv instance
        Q (ndarray): contains the Q-table
        episodes (int, optional): total number of episodes to train over.
                                  Defaults to 5000.
        max_steps (int, optional): maximum number of steps per episode.
                                   Defaults to 100.
        alpha (float, optional): learning rate. Defaults to 0.1.
        gamma (float, optional): discount rate. Defaults to 0.99.
        epsilon (int, optional): initial threshold for epsilon greedy.
                                 Defaults to 1.
        min_epsilon (float, optional): minimum value that epsilon should decay
                                       to. Defaults to 0.1.
        epsilon_decay (float, optional): decay rate for updating epsilon.
                                         Defaults to 0.05.
    Returns:
        Q: updated Q-table
        total_rewards: list containing the rewards per episode
    """
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, inf = env.step(action)
            Q[state, action] = Q[state, action] + alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
            if done is True:
                if reward == 0.0:
                    total_rewards = -1
                total_rewards += reward
                break
            total_rewards += reward
        epsilon = min_epsilon + (1 - min_epsilon) * \
            np.exp(-epsilon_decay * episode)
        rewards.append(total_rewards)
    return Q, rewards
