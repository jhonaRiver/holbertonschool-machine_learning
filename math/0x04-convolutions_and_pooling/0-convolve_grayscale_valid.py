#!/usr/bin/env python3
"""
module convolve_grayscale_valid
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images
    Args:
        images (ndarray): contains multiple grayscale images
        kernel (ndarray): contains the kernel for the convolution
    Returns:
        ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    dim = (m, h - kh + 1, w - kw + 1)
    out = np.zeros(dim)
    for i in range(dim[1]):
        for j in range(dim[2]):
            x = i + kh
            y = j + kw
            M = images[:, i:x, j:y]
            out[:, i, j] = np.tensordot(M, kernel)
    return out
