#!/usr/bin/env python3
"""
module mat_mul
"""


def mat_mul(mat1, mat2):
    """
    performs matrix multiplication
    Args:
        mat1 (int/float): 2D matrix containing ints/floats
        mat2 (int/float): 2D matrix containing ints/floats
    """
    product = []
    if len(mat1[0]) != len(mat2):
        return None
    for idx in range(len(mat1)):
        product.append([])
        for j in range(len(mat2[0])):
            product[idx].append(0)
            for k in range(len(mat2)):
                product[idx][j] += mat1[idx][k] * mat2[k][j]
    return product
