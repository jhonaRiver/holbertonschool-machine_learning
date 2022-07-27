#!/usr/bin/env python3
"""
module add_arrays
"""


def add_arrays(arr1, arr2):
    """
    adds two arrays element-wise
    Args:
        arr1 (int/float): list of ints/floats to be added
        arr2 (int/float): list of ints/floats to be added
    """
    if len(arr1) != len(arr2):
        return None
    else:
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
