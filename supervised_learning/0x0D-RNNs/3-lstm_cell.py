#!/usr/bin/env python3
"""Class LSTMCell."""
import numpy as np


class LSTMCell:
    """Represent an LSTM unit."""

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (ndarray): contains the previous hidden state
            c_prev (ndarray): contains the previous cell state
            x_t (ndarray): contains the data input for the cell
        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
