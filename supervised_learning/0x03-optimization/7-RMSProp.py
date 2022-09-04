#!/usr/bin/env python3
"""
module update_variables_RMSProp
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm
    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp weight
        epsilon (float): small number to avoid division by zero
        var (ndarray): contains the variable to be updated
        grad (ndarray): contains the gradient of var
        s (ndarray): previous second moment of var
    Returns:
        updated variable and the new moment
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    updt = var - alpha * grad / (s ** (1/2) + epsilon)
    return updt, s
