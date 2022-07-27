#!/usr/bin/env python3
"""
module cat_matrices2D
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    Args:
        mat1 (int/float): 2D matrix containing ints/floats
        mat2 (int/float): 2D matrix containing ints/floats
        axis (int, optional): axis to concatenate at. Defaults to 0.
    """
    newMatrix = []
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat1:
            newMatrix.append(row)
        for row in mat2:
            newMatrix.append(row)
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for row1, row2 in zip(mat1, mat2):
            newMatrix.append(row1 + row2)
    return newMatrix