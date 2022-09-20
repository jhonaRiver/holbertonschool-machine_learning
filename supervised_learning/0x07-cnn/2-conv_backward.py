#!/usr/bin/env python3
"""
module conv_backward
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network
    Args:
        dZ (ndarray): contains the partial derivatives with respect to the
                      unactivated output of the convolutional layer
        A_prev (ndarray): contains the output of the previous layer
        W (ndarray): contains the kernels for the convolution
        b (ndarray): contains the biases applied to the convolution
        padding (str, optional): indicates the type of padding used. Defaults
                                 to "same".
        stride (tuple, optional): contains the strides for the convolution.
                                  Defaults to (1, 1).
    Returns:
        partial derivatives with respect to the previous layer, the kernels,
        and the biases
    """
