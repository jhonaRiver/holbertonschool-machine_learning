#!/usr/bin/env python3
"""Class GaussianProcess."""
import numpy as np


class GaussianProcess:
    """Represent a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor.

        Args:
            X_init (ndarray): represents the inputs already sampled with the
                              black-box function
            Y_init (ndarray): represents the outputs of the black-box function
                              for each input in X_init
            l (int, optional): length parameter for the kernel. Defaults to 1.
            sigma_f (int, optional): standard deviation given to the output of
                                     the black-box function. Defaults to 1.
        """

    def kernel(self, X1, X2):
        """
        Calculate the covariance kernel matrix between two matrices.

        Args:
            X1 (ndarray): matrix 1
            X2 (ndarray): matrix 2
        Returns:
            convariance kernel matrix
        """

    def predict(self, X_s):
        """
        Predict the mean and standard deviation of points in a Gaussian\
            process.

        Args:
            X_s (ndarray): contains all of the points whose mean and standard
                           deviation should be calculated
        Returns:
            mu: contains the mean for each point in X_s
            sigma: contains the variance for each point in X_s
        """

    def update(self, X_new, Y_new):
        """
        Update a Gaussian Process.

        Args:
            X_new (ndarray): represents the new sample point
            Y_new (ndarray): represents the new sample function value
        """
