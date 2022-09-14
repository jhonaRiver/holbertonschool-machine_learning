#!/usr/bin/env python3
"""
module convolve_grayscale_padding
"""


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding
    Args:
        images (ndarray): contains multiple grayscale images
        kernel (ndarray): contains the kernel for the convolution
        padding (tuple): contains the height and weight of the padding
    Returns:
        ndarray containing the convolved images
    """
