#!/usr/bin/env python3
"""Module class MultiNormal."""
import numpy as np


class MultiNormal:
    """Represent a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Class constructor.

        Args:
            data (ndarray): contains the data set
        """

    def pdf(self, x):
        """
        Calculate the PDF at a data point.

        Args:
            x (ndarray): contains the data point whose PDF should be calculated
        Returns:
            value of the PDF
        """
