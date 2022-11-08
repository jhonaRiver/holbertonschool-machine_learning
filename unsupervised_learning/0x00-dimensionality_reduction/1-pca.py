#!/usr/bin/env python3
"""Module pca."""
import numpy as np


def pca(X, ndim):
    """
    Perform PCA on a dataset.

    Args:
        X (ndarray): contains dataset
        ndim (int): new dimensionality of the transformed X
    Returns:
        transformed version of X
    """
