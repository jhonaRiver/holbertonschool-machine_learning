#!/usr/bin/env python3
"""
module conv_forward
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional layer of a neural network
    Args:
        A_prev (ndarray): contains the output of the previous layer
        W (ndarray): contains the kernels for the convolution
        b (ndarray): contains the biases applied to the convolution
        activation (function): activation function applied to the convolution
        padding (str, optional): indicates the type of padding used. Defaults
                                 to "same".
        stride (tuple, optional): contains the stride for the convolution.
                                  Defaults to (1, 1).
    Returns:
        output of the convolutional layer
    """
