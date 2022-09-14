#!/usr/bin/env python3
"""
module test_model
"""


def test_model(network, data, labels, verbose=True):
    """
    tests a neural network
    Args:
        network (keras): model to test
        data (ndarray): input data to test the model with
        labels (ndarray): correct one-hot labels of data
        verbose (boolean, optional): determines if output should be printed
                                     during the testing process. Defaults to
                                     True.
    Returns:
        loss and accuracy of the model with the testing data
    """
    return network.evaluate(data, labels, verbose=verbose)
