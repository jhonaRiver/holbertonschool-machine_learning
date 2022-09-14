#!/usr/bin/env python3
"""
module predict
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network
    Args:
        network (keras): model to make the prediction with
        data (ndarray): input data to make the prediction with
        verbose (boolean, optional): determines if output should be printed
                                     during the prediction process. Defaults
                                     to False.
    Returns:
        prediction for the data
    """
    return network.predict(data, verbose=verbose)
