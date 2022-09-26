#!/usr/bin/env python3
"""
module identity_block
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    builds an identity block as described in Deep Residual Learning for Image
    Recognition(2015)
    Args:
        A_prev (ndarray): output from the previous layer
        filters (tuple or list): contains F11, F3, F12
    Returns:
        activated output of the identity block
    """
