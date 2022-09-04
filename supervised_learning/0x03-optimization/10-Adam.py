#!/usr/bin/env python3
"""
module create_Adam_op
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    Args:
        loss (tensorflow): loss of the network
        alpha (float): learning rate
        beta1 (float): weight used for the first moment
        beta2 (float): weight used for the second moment
        epsilon (float): small number to avoid division by zero
    Returns:
        Adam optimization operation
    """
    op = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return op.minimize(loss)
