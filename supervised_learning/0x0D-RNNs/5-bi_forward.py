#!/usr/bin/env python3
"""Class BidirectionalCell."""
import numpy as np


class BidirectionalCell:
    """Represent a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden states
            o (int): dimensionality of the outputs
        """

    def forward(self, h_prev, x_t):
        """
        Calculate the hidden state in the forward direction for one time step.

        Args:
            h_prev (ndarray): contains the previous hidden state
            x_t (ndarray): contains the data input for the cell
        Returns:
            h_next: next hidden state
        """
