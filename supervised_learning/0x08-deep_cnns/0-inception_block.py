#!/usr/bin/env python3
"""
module inception_block
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds an inception block as described in Going Deeper with
    Convolutions(2014)
    Args:
        A_prev (ndarray): output from the previous layer
        filters (tuple or list): contains F1, F3R, F3, F5R, F5, FPP
    Returns:
        concatenated output of the inception block
    """
