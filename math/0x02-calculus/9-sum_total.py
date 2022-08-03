#!/usr/bin/env python3
"""
module summation_i_squared
"""


def summation_i_squared(n):
    """
    calculates summation
    Args:
        n (int): stopping condition

    Returns:
        int: sum
    """
    if type(n) != int:
        return None
    if n == 1:
        return n
    return ((n ** 2) + summation_i_squared(n - 1))