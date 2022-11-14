#!/usr/bin/env python3
"""Module pdf."""
import numpy as np


def pdf(X, m, S):
    """
    Calculate the probability density function of a Gaussian distribution.

    Args:
        X (ndarray): contains the data points
        m (ndarray): contains the mean
        S (ndarray): contains the covariance
    Returns:
        P or None
    """
