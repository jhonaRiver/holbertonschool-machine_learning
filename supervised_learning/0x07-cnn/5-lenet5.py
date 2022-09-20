#!/usr/bin/env python3
"""
module lenet5
"""
import tensorflow.keras as K


def lenet5(X):
    """
    builds a modified version of the LeNet-5 architecture using keras
    Args:
        X (input): contains the input images for the network
    Returns:
        K.model compiled to use Adam optimization and accuracy metrics
    """
