#!/usr/bin/env python3
"""
module convolve_grayscale_valid
"""


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images
    Args:
        images (ndarray): contains multiple grayscale images
        kernel (ndarray): contains the kernel for the convolution
    Returns:
        ndarray containing the convolved images
    """
