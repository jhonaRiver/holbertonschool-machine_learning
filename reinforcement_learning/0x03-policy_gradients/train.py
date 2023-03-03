#!/usr/bin/env python3
"""Module train."""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Implement a full training.

    Args:
        env (_type_): initial environment
        nb_episodes (_type_): number of episodes used for training
        alpha (float, optional): learning rate. Defaults to 0.000045.
        gamma (float, optional): discount factor. Defaults to 0.98.
        show_result (bool, optional): determines wether to show the result.
                                      Defaults to False.
    Returns:
        all values of the score
    """
    weight = np.random.rand(4, 2)
    all_scores = []
    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        gradients = []
        rewards = []
        sum_rewards = 0
        while True:
            if show_result and (episode % 1000 == 0):
                env.render()
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, info = env.step(action)
            gradients.append(gradient)
            rewards.append(reward)
            sum_rewards += reward
            if done:
                break
            state = next_state[None, :]
        for i in range(len(gradients)):
            weight += (alpha * gradients[i] * sum([r * (gamma ** r)
                       for t, r in enumerate(rewards[i:])]))
        all_scores.append(sum_rewards)
        print("{}: {}".format(episode, sum_rewards), end="\r", flush=False)
    return all_scores
