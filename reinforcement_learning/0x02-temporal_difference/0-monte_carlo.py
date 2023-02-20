#!/usr/bin/env python3
"""Module monte_carlo."""
# import gym
# import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Perform the Monte Carlo algorithm.

    Args:
        env (obj): openAI environment instance
        V (ndarray): contains the value estimate
        policy (func): takes in a state and returns the next action to take
        episodes (int, optional): total number of episodes to train over.
                                  Defaults to 5000.
        max_steps (int, optional): maximum number of steps per episode.
                                   Defaults to 100.
        alpha (float, optional): learning rate. Defaults to 0.1.
        gamma (float, optional): discount rate. Defaults to 0.99.
    Returns:
        V: updated value estimate
    """
    # Loop for the given number of episodes
    for i in range(episodes):
        # Initialize episode list and state
        episode = []
        state = env.reset()

        # Loop for the given number of steps
        for j in range(max_steps):
            # Get the action to take from the policy
            action = policy(state)

            # Take the action and observe the next state, reward, and done flag
            next_state, reward, done, _ = env.step(action)

            # Add the (state, action, reward) tuple to the episode
            episode.append((state, action, reward))

            # Update the state
            state = next_state

            # If the episode is done, break out of the loop
            if done:
                break

        # Loop over the episode in reverse order to calculate the returns and
        # update the value estimate
        G = 0
        for state, action, reward in reversed(episode):
            # Calculate the return
            G = gamma * G + reward

            # Update the value estimate
            V[state] += alpha * (G - V[state])

    return V
