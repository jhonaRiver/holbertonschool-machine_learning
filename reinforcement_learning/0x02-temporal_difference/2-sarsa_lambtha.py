#!/usr/bin/env python3
"""Module sarsa_lambtha."""
import gym
import numpy as np


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
    # Loop for the given number of episodes
    for i in range(episodes):
        # Initialize eligibility trace, state, and action
        E = np.zeros(Q.shape)
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        # Loop for the given number of steps
        for j in range(max_steps):
            # Take the action and observe the next state, reward, and done flag
            next_state, reward, done, _ = env.step(action)

            # Get the next action from the next state using epsilon greedy
            next_action = epsilon_greedy(Q, next_state, epsilon)

            # Calculate the TD error
            delta = reward + gamma * \
                Q[next_state][next_action] - Q[state][action]

            # Update the eligibility trace
            E[state][action] += 1

            # Update the Q table and eligibility trace for all state-action
            # pairs
            for s in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    Q[s][a] += alpha * delta * E[s][a]
                    E[s][a] *= gamma * lambtha * (action == a and state == s)

            # Update the state and action
            state = next_state
            action = next_action

            # If the episode is done, break out of the loop
            if done:
                break

        # Decay the epsilon value
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q


def epsilon_greedy(Q, state, epsilon):
    """Epsilon greedy."""
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action
