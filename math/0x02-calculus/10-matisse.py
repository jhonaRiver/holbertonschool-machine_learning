#!/usr/bin/env python3
"""
module poly_derivative
"""


def poly_derivative(poly):
    """
    calculates the derivative of a polynomial
    Args:
        poly (list): list of coefficients representing a polynomial

    Returns:
        list: new list of coefficients representing the derivative of the
        polynomial
    """    
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    if not any(isinstance(val, int) for val in poly):
        return None
    der_poly = []
    for power, coefficient in enumerate(poly):
        if power == 0:
            der_poly.append(0)
        if power == 1:
            der_poly = []
        der_poly.append(power * coefficient)
    return der_poly
