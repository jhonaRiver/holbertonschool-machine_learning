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
    if not isinstance(n, int) or n < 1:
        return None
    elif n == 1:
        return 1
    return int(n ** 2) + int(summation_i_squared(n - 1))
