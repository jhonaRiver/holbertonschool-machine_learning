#!/usr/bin/env python3
"""
module save_weights and load_weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves a model's weights
    Args:
        network (keras): model whose weights should be saved
        filename (str): path of the file that the weights should be saved to
        save_format (str, optional): format in which the weights should be
                                     saved. Defaults to h5.
    Returns:
        None
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    loads a model's weights
    Args:
        network (keras): model to which the weights should be loaded
        filename (str): path of the file that the weights should be loaded
                        from
    Returns:
        None
    """
    network.load_weights(filename)
    return None
