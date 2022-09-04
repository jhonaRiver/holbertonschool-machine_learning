#!/usr/bin/env python3
"""
module update_variables_Adam
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using the Adam optimization algorithm
    Args:
        alpha (float): learning rate
        beta1 (float): weight used for the first moment
        beta2 (float): weight used for the second moment
        epsilon (float): small number to avoid division by zero
        var (ndarray): contains the variable to be updated
        grad (ndarray): contains the gradient of var
        v (float): previous first moment of var
        s (float): previous second moment of var
        t (int): time step used for bias correction
    Returns:
        updated variable, new first moment, new second moment
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    v_corr = v / (1 - (beta1 ** t))
    s_corr = s / (1 - (beta2 ** t))
    var = var - alpha * v_corr / (s_corr ** (1/2) + epsilon)
    return var, v, s
