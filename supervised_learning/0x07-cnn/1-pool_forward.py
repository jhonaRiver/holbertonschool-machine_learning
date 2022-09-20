#!/usr/bin/env python3
"""
module pool_forward
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network
    Args:
        A_prev (ndarray): contains the output of the previous layer
        kernel_shape (tuple): contains the size of the kernel for the pooling
        stride (tuple, optional): contains the strides for the pooling.
                                  Defaults to (1, 1).
        mode (str, optional): indicates whether to perform maximum or average
                              pooling. Defaults to 'max'.
    Returns:
        output of the pooling layer
    """
