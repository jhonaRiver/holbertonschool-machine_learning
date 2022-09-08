#!/usr/bin/env python3
"""
module l2_reg_create_layer
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a tensorflow layer that includes L2 regularization
    Args:
        prev (tensor): contains the output of the previous layer
        n (int): number of nodes the new layer should contain
        activation (func): activation function that should be used on the
                           layer
        lambtha (float): L2 regularization parameter
    Returns:
        output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode=("fan_avg"))
    kernel_regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=initializer, name='layer',
                                  kernel_regularizer=kernel_regularizer)
    return layer(prev)
