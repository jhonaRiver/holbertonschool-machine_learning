#!/usr/bin/env python3
"""Module class MultiNormal."""
import numpy as np


class MultiNormal:
    """Represent a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Class constructor.

        Args:
            data (ndarray): contains the data set
        """
        if type(data) is not np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        d, n = data.shape
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        X_mean = data - self.mean
        self.cov = np.matmul(X_mean, X_mean.T) / (n - 1)

    def pdf(self, x):
        """
        Calculate the PDF at a data point.

        Args:
            x (ndarray): contains the data point whose PDF should be calculated
        Returns:
            value of the PDF
        """
