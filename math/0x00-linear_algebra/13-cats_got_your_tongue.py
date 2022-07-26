#!/usr/bin/env python3
"""
module np_cat
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    Args:
        mat1 (int): ndarray to concatenate
        mat2 (int): ndarray to concatenate
        axis (int, optional): axis to concatenate. Defaults to 0.
    """
    return np.append(mat1, mat2, axis)