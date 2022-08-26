#!/usr/bin/env python3
"""
module calculate_loss
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_loss(y, y_pred):
    """
    calculates the softmax cross-entropy loss of a prediction
    Args:
        y (tensor): placeholder for the labels of the input data
        y_pred (tensor): contains the network's predictions

    Returns:
        tensor: tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
