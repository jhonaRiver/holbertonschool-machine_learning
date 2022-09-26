#!/usr/bin/env python3
"""
module transition_layer
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer as described in Densely Connected Convolutional
    Networks
    Args:
        X (ndarray): output from the previous layer
        nb_filters (int): represents the number of filters in X
        compression (float): compression factor for the transition layer
    Returns:
        output of the transition layer and the number of filters within the
        output
    """
