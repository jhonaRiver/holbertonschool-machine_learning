#!/usr/bin/env python3
"""Module td_lambtha."""
import gym
import numpy as np


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
    # Loop for the given number of episodes
    for i in range(episodes):
        # Initialize eligibility trace and state
        E = np.zeros(V.shape)
        state = env.reset()

        # Loop for the given number of steps
        for j in range(max_steps):
            # Get the action to take from the policy
            action = policy(state)

            # Take the action and observe the next state, reward, and done flag
            next_state, reward, done, _ = env.step(action)

            # Calculate the TD error
            delta = reward + gamma * V[next_state] - V[state]

            # Update the eligibility trace
            E[state] += 1

            # Loop over all states and update the value estimate and
            # eligibility trace
            for s in range(env.observation_space.n):
                V[s] += alpha * delta * E[s]
                E[s] *= gamma * lambtha

            # Update the state
            state = next_state

            # If the episode is done, break out of the loop
            if done:
                break

    return V
