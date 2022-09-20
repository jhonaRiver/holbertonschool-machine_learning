#!/usr/bin/env python3
"""
module lenet5
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    builds a modified version of the Lenet-5 architecture using tensorflow
    Args:
        x (placeholder): contains the input images for the network
        y (placeholder): contains the one-hot labels for the network
    Returns:
        tensor for the softmax activated output, training operation that
        utilizes Adam optimization, tensor for the loss of the network, tensor
        for the accuracy of the network
    """
