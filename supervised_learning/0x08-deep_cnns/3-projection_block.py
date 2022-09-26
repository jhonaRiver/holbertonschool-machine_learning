#!/usr/bin/env python3
"""
module projection_block
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    builds a projection block as described in Deep Residual Learning for Image
    Recognition(2015)
    Args:
        A_prev (ndarray): output from the previous layer
        filters (tuple or list): contains F11, F3, F12
        s (int, optional): stride of the first convolution in both the main
                           path and the shortcut connection. Defaults to 2.
    Returns:
        activated output of the projection block
    """
