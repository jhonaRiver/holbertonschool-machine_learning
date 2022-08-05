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
    if poly is None or poly == [] or type(poly) != list:
        return None
    if type(C) is int or type(C) is float:
        if poly == [0]:
            return [C]
        if C % 1 == 0:
            C = int(C)
        integral = [C]
        for power in range(len(poly)):
            if type(poly[power]) != int and type(poly[power]) != float:
                return None
            if (poly[power] / (power + 1)) % 1 == 0:
                integral.append(int(poly[power] / (power + 1)))
            else:
                integral.append(poly[power] / (power + 1))
        for power in range(len(integral) - 1, 0, -1):
            if integral[power] == 0:
                integral.pop()
            else:
                break
        return integral
    return None
