#!/usr/bin/env python3
"""Class RNNCell."""
import numpy as np


class RNNCell:
    """Represent a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (ndarray): contains the previous hidden state
            x_t (ndarray): contains the data input for the cell
        Returns:
            h_next: next hidden state
            y: output of the cell
        """
