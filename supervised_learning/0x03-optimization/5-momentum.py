#!/usr/bin/env python3
"""
module update_variables_momentum
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent with momentum optimization
    algorithm
    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        var (ndarray): contains the variable to be updated
        grad (ndarray): contains the gradient of var
        v (ndarray): previous first moment of var
    Returns:
        updated variable and the new moment
    """
    momentum = beta1 * v + (1 - beta1) * grad
    update = var - alpha * momentum
    return momentum, update
