#!/usr/bin/env python3
"""
module convolve_grayscale
"""
import numpy as np


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
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride[0], stride[1]
    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + (kh % 2 == 0))
        pw = int((((w - 1) * sw + kw - w) / 2) + (kw % 2 == 0))
    else:
        ph, pw = padding[0], padding[1]
    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)
    dim = (m, oh, ow)
    out = np.zeros(dim)
    padded = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant', constant_values=0)
    for i in range(dim[1]):
        for j in range(dim[2]):
            x = (i * sh) + kh
            y = (j * sw) + kw
            M = padded[:, (i * sh):x, (j * sw):y]
            out[:, i, j] = np.tensordot(M, kernel)
    return out
