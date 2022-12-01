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
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculate the covariance kernel matrix between two matrices.

        Args:
            X1 (ndarray): matrix 1
            X2 (ndarray): matrix 2
        Returns:
            convariance kernel matrix
        """
        sqdist1 = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        sqdist2 = 2 * np.dot(X1, X2.T)
        sqdist = sqdist1 - sqdist2
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

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
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = np.reshape(mu_s, -1)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        cov_s = cov_s.diagonal()
        return mu_s, cov_s
