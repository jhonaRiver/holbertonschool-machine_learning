#!/usr/bin/env python3
"""
module dense_block
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in Densely Connected Convolutional
    Networks
    Args:
        X (ndarray): output from the previous layer
        nb_filters (int): represents the number of filters in X
        growth_rate (int): growth rate for the dense block
        layers (int): number of layers in the dense block
    Returns:
        concatenated output of each layer within the Dense Block and the
        number of filters within the concatenated outputs
    """
