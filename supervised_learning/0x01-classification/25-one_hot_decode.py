#!/usr/bin/env python3
"""
module one_hot_decode
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    convertsa one-hot matrix into a vector of labels
    Args:
        one_hot (ndarray): one-hot encoded matrix

    Returns:
        ndarray: contains the numeric labels for each example or None on
                 failure
    """
    if type(one_hot) is np.ndarray and len(one_hot.shape) == 2:
        return np.argmax(one_hot, axis=0)
    return None
