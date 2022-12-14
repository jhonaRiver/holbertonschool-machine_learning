#!/usr/bin/env python3
"""Module rnn."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN.

    Args:
        rnn_cell (object): used for the forward propagation
        X (ndarray): data to be used
        h_0 (ndarray): initial hidden state
    Returns:
        H: contains all of the hidden states
        Y: contains all of the outputs
    """
