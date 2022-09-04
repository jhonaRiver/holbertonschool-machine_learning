#!/usr/bin/env python3
"""
module create_RMSProp_op
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm
    Args:
        loss (tensorflow): loss of the network
        alpha (float): learning rate
        beta2 (float): RMSProp weight
        epsilon (float): small number to avoid division by zero
    Returns:
        RMSProp optimization operation
    """
    op = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                   epsilon=epsilon)
    step_op = op.minimize(loss)
    return step_op
