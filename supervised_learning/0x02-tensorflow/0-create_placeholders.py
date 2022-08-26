#!/usr/bin/env python3
"""
module create_placeholders
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    """
    function that returns two placeholders
    Args:
        nx (int): number of feature columns in our data
        classes (int): number of classes in our classifier

    Returns:
        tensor: placeholder for the input data to the neural network,
                placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
