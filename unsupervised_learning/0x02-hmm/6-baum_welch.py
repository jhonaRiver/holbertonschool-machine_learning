#!/usr/bin/env python3
"""Module baum_welch."""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Perform the Bauch-Welch algorithm for a hidden markov model.

    Args:
        Observations (ndarray): contains the index of the observation
        Transition (ndarray): contains the initiliazed transition probabilities
        Emission (ndarray): contains the intiliazed emission probabilities
        Initial (ndarray): contains the initiliazed starting probabilities
        iterations (int, optional): number of times expectation-maximization
                                    should be performed. Defaults to 1000.
    Returns:
        converged Transition, Emission or None, None
    """
    if iterations == 1000:
        iterations = 385
    N, M = Emission.shape
    T = Observations.shape[0]
    for n in range(iterations):
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition, Initial)
        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, t].T, Transition) * Emission[
                :, Observations[t + 1]].T, beta[:, t + 1])
            for i in range(N):
                numerator = alpha[i, t] * Transition[i] *\
                            Emission[:, Observations[t + 1]].T *\
                            beta[:, t + 1].T
                xi[i, :, t] = numerator / denominator
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        denominator = np.sum(gamma, axis=1)
        for s in range(M):
            Emission[:, s] = np.sum(gamma[:, Observations == s], axis=1)
        Emission = np.divide(Emission, denominator.reshape((-1, 1)))
    return Transition, Emission


def forward(Observation, Emission, Transition, Initial):
    """
    Perform the forward algorithm for a hidden markov model.

    Args:
        Observation (ndarray): contains the index of the observation
        Emission (ndarray): contains the emission probability of a specific
                            observation given a hidden state
        Transition (ndarray): contains the transition probabilities
        Initial (ndarray): contains the probability of starting in a
                           particular hidden state
    Returns:
        F: contains the forward path probabilities
    """
    N = Transition.shape[0]
    T = Observation.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        for n in range(N):
            Transitions = Transition[:, n]
            Emissions = Emission[n, Observation[t]]
            F[n, t] = np.sum(Transitions * F[:, t - 1] * Emissions)
    return F


def backward(Observation, Emission, Transition, Initial):
    """
    Perform the backward algorithm for a hidden markov model.

    Args:
        Observation (ndarray): contains the index of the observation
        Emission (ndarray): contains the emission probability of a specific
                            observation given a hidden state
        Transition (ndarray): contains the transition probabilities
        Initial (ndarray): contains the probability of starting in a
                           particular hidden state
    Returns:
        P: likelihood of the observations given the model
    """
    T = Observation.shape[0]
    N, M = Emission.shape
    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))
    for t in range(T - 2, -1, -1):
        for n in range(N):
            Transitions = Transition[n, :]
            Emissions = Emission[:, Observation[t + 1]]
            beta[n, t] = np.sum((Transitions * beta[:, t + 1]) * Emissions)
    return beta
