#!/usr/bin/env python3
"""
module batch_norm
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network using batch
    normalization
    Args:
        Z (ndarray): should be normalized
        gamma (ndarray): contains the scales used for batch normalization
        beta (ndarray): contains the offsets used for batch normalization
        epsilon (float): small number used to avoid division by zero
    Returns:
        normalized Z matrix
    """
    m = Z.shape[0]
    mean = 1 / m * np.sum(Z, axis=0)
    variance = 1 / m * np.sum((Z - mean) ** 2, axis=0)
    Z_norm = (Z - mean) / (np.sqrt(variance + epsilon))
    return (gamma * Z_norm) + beta
