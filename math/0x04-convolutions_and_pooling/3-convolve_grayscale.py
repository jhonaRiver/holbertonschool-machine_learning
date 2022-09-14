#!/usr/bin/env python3
"""
module convolve_grayscale
"""


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images
    Args:
        images (ndarray): contains multiple grayscale images
        kernel (ndarray): contains the kernel for the convolution
        padding (str, optional): contains the padding of the image. Defaults
                                 to 'same'.
        stride (tuple, optional): contains the stride of the image. Defaults
                                  to (1, 1).
    Returns:
        ndarray containing the convolved images
    """
