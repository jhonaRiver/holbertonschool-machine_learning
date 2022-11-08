#!/usr/bin/env python3
"""Module pca."""
import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on a dataset.

    Args:
        X (ndarray): contains dataset
        var (float, optional): fraction of the variance. Defaults to 0.95.
    Returns:
        weights matrix
    """
