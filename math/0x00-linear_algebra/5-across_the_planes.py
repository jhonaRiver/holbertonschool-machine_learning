#!/usr/bin/env python3
"""
module add_matrices2D
"""


def add_matrices2D(mat1, mat2):
    """
    adds two matrices element-wise
    Args:
        mat1 (int/float): 2D matrix containing ints/floats
        mat2 (int/float): 2D matrix containing ints/floats
    """
    newMatrix = []
    if len(mat1[0]) != len(mat2[0]):
        return None
    for idx in range(len(mat1)):
        sum = []
        for idx2 in range(len(mat1)):
            sum.append((mat1[idx][idx2] + mat2[idx][idx2]))
        newMatrix.append(sum)
    return newMatrix