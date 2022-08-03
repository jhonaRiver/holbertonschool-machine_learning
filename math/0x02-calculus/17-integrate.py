#!/usr/bin/env python3
"""
module poly_integral
"""


def poly_integral(poly, C=0):
    """
    calculates the integral of a polynomial
    Args:
        poly (list): list of coefficients representing a polynomial
        C (int, optional): integer representing the integration constant
                           Defaults to 0.

    Returns:
        list: new list of coefficients representing the integral of the
              polynomial
    """
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    if not isinstance(C, (int, float)):
        return None
    if not any(isinstance(val, (int, float)) for val in poly):
        return None
    if isinstance(C, float) and C.is_integer:
        C = int(C)
    integral = [C]
    for power, coefficient in enumerate(poly):
        if (coefficient % (power + 1)) == 0:
            newCoefficient = coefficient // (power + 1)
        else:
            newCoefficient = coefficient / (power + 1)
        integral.append(newCoefficient)
    return integral
