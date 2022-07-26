#!/usr/bin/env python3
"""
module np_elementwise
"""


def np_elementwise(mat1, mat2):
    """
    performs element-wise addition, subtraction, multiplication and division
    Args:
        mat1 (int): ndarray to perform operation
        mat2 (int): ndarray to perform operation
    """
    return (mat1+mat2, mat1-mat2, mat1*mat2, mat1/mat2)