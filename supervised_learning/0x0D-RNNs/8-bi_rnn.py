#!/usr/bin/env python3
"""Module bi_rnn."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation for a bidirectional RNN.

    Args:
        bi_cell (object): will be used for the forward propagation
        X (ndarray): data to be used
        h_0 (ndarray): initial hidden state in the forward direction
        h_t (ndarray): initial hidden state in the backward direction
    Returns:
        H: contains all of the concatenated hidden states
        Y: contains all of the outputs
    """
