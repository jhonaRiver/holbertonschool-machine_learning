#!/usr/bin/env python3
"""
module densenet121
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks
    Args:
        growth_rate (int, optional): growth rate. Defaults to 32.
        compression (float, optional): compression factor. Defaults to 1.0.
    Returns:
        keras model
    """
