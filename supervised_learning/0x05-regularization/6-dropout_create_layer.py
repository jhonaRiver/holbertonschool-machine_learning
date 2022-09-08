#!/usr/bin/env python3
"""
module dropout_create_layer
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout
    Args:
        prev (tensor): contains the output of the previous layer
        n (int): number of nodes the new layer should contain
        activation (func): activation function that should be used on the
                           layer
        keep_prob (float): probability that a node will be kept
    Returns:
        output of the new layer
    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=dropout)
    return layer(prev)
