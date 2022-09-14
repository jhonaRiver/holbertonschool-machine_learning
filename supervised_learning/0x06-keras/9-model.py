#!/usr/bin/env python3
"""
module save_model and load_model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model
    Args:
        network (keras): model to save
        filename (str): path of the file that the model should be saved to
    Returns:
        None
    """
    K.models.save_model(model=network, filepath=filename)
    return None


def load_model(filename):
    """
    loads an entire model
    Args:
        filename (str): path of the file that the model should be loaded from
    Returns:
        loaded model
    """
    return K.models.load_model(filepath=filename)
