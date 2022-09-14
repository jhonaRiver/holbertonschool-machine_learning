#!/usr/bin/env python3
"""
module convolve_grayscale_same
"""


def convolve_grayscale_same(images, kernel):
    """
    performs a same convolution on grayscale images
    Args:
        images (ndarray): contains multiple grayscale images
        kernel (ndarray): contains the kernel for the convolution
    Returns:
        ndarray containing the convolved images
    """
