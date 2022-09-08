#!/usr/bin/env python3
"""
module l2_reg_cost
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization
    Args:
        cost (tensor): contains the cost of the network without L2
                       regularization
    Returns:
        tensor containing the cost of the network accounting for L2
        regularization
    """
    return cost + tf.losses.get_regularization_losses()
