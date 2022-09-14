#!/usr/bin/env python3
"""
module save_config and load_config
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model's configuration in JSON format
    Args:
        network (keras): model whose configuration should be saved
        filename (str): path of the file that the configuration should be
                        saved to
    Returns:
        None
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())
    f.close()
    return None


def load_config(filename):
    """
    loads a model with a specific configuration
    Args:
        filename (str): path of the file containing the model's configuration
                        in JSON format
    Returns:
        loaded model
    """
    with open(filename, 'r') as f:
        model = K.models.model_from_json(f.read())
    f.close()
    return model
