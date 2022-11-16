#!/usr/bin/env python3
"""Module kmeans."""
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means on a dataset.

    Args:
        X (ndarray): contains the dataset
        k (int): number of clusters
    Returns:
        C, clss
    """
    k_mean = sklearn.cluster.KMeans(n_clusters=k)
    k_mean.fit(X)
    clss = k_mean.labels_
    C = k_mean.cluster_centers_
    return C, clss
