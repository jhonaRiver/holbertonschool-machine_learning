#!/usr/bin/env python3
"""
module pool
"""


def pool(images, kernel_shape, stride, mode='max'):
    """
    performs pooling on images
    Args:
        images (ndarray): contains multiple images
        kernel_shape (tuple): contains the kernel shape for the pooling
        stride (tuple): contains the stride of the image
        mode (str, optional): type of pooling. Defaults to 'max'.
    Returns:
        ndarray containing the pooled images
    """
