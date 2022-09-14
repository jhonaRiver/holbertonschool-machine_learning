#!/usr/bin/env python3
"""
module convolve
"""


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    performs a convolution on images using multiple kernels
    Args:
        images (ndarray): contains multiple images
        kernels (ndarray): contains the kernels for the convolution
        padding (str, optional): contains the padding of the image. Defaults
                                 to 'same'.
        stride (tuple, optional): contains the stride of the image. Defaults
                                  to (1, 1).
    Returns:
        ndarray containing the convolved images
    """
