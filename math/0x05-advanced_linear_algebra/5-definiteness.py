#!/usr/bin/env python3
"""Module definiteness."""
import numpy as np


def definiteness(matrix):
    """
    Calculate the definiteness of a matrix.

    Args:
        matrix (ndarray): whose definiteness should be calculated
    Returns:
        positive definite, positive semi-definite, negative semi-definite,
        negative definite, indefinite or None
    """
