#!/usr/bin/env python3
"""Module  viterbi."""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculate the most likely sequence of hidden states for a hidden markov\
        model.

    Args:
        Observation (ndarray): contains the index of the observation
        Emission (ndarray): contains the emission probability of a specific
                            observation given a hidden state
        Transition (ndarray): contains the transition probabilities
        Initial (ndarray): contains the probability of starting in a
                           particular hidden state
    Returns:
        path: contains the most likely sequence of hidden states
        P: probability of obtaining the path sequence
        None: failure
    """
