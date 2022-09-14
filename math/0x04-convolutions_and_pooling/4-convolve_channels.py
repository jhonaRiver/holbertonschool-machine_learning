#!/usr/bin/env python3
"""
module convolve_channels
"""


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on images with channels
    Args:
        images (ndarray): contains multiple images
        kernel (ndarray): contains the kernel for the convolution
        padding (str, optional): contains the padding of the image. Defaults
                                 to 'same'.
        stride (tuple, optional): contains the stride of the image. Defaults
                                  to (1, 1).
    Returns:
        ndarray containing the convolved images
    """
