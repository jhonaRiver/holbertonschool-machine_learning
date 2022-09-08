#!/usr/bin/env python3
"""
module l2_reg_cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization
    Args:
        cost (float): cost of the network without L2 regularization
        lambtha (float): regularization parameter
        weights (dict): dictionary of the weights and biases of the neural
                        network
        L (int): number of layers in the neural network
        m (int): number of data points used
    Returns:
        cost of the network accounting for L2 regularization
    """
    l2 = 0
    for idx in range(1, L+1):
        l2 += np.linalg.norm(weights['W' + str(idx)])
    return cost + (l2 * lambtha / (2 * m))
