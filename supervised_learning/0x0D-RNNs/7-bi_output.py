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
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculate the hidden state in the forward direction for one time step.

        Args:
            h_prev (ndarray): contains the previous hidden state
            x_t (ndarray): contains the data input for the cell
        Returns:
            h_next: next hidden state
        """
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)
        h_next = np.tanh((h_x.T @ self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculate the hidden state in the backward direction for one time step.

        Args:
            h_next (ndarray): contains the next hidden state
            x_t (ndarray): contains the data input for the cell
        Returns:
            h_pev: previous hidden state
        """
        h_x = np.concatenate((h_next.T, x_t.T), axis=0)
        h_prev = np.tanh((h_x.T @ self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculate all outputs for the RNN.

        Args:
            H (ndarray): contains the concatenated hidden states from both
                         directions, excluding their initialized states
        Returns:
            Y: outputs
        """
        t, m, _ = H.shape
        time_step = range(t)
        o = self.by.shape[1]
        Y = np.zeros((t, m, o))
        for ts in time_step:
            y_pred = self.softmax((H[ts] @ self.Wy) + self.by)
            Y[ts] = y_pred
        return Y

    def softmax(slef, x):
        """Softmax function."""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
