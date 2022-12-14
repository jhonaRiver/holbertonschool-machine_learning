#!/usr/bin/env python3
"""Module deep_rnn."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Perform forward propagation for a deep RNN.

    Args:
        rnn_cells (list): contains RNNCell instances that will be used for the
                          forward propagation
        X (ndarray): data to be used
        h_0 (ndarray): initial hidden state
    Returns:
        H: contains all of the hidden states
        Y: contains all of the outputs
    """
