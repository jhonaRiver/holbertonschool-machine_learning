#!/usr/bin/env python3
"""Module expectation."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X (ndarray): contains the data set
        pi (ndarray): contains the priors
        m (ndarray): contains the centroid means
        S (ndarray): contains the covariance matrices
    Returns:
        g, l or None, none
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
        return None, None
    if k != m.shape[0] or k != S.shape[0]:
        return None, None
    centroids_mean = m
    covariance_mat = S
    gauss_components = np.zeros((k, n))
    for i in range(k):
        likelihood = pdf(X, centroids_mean[i], covariance_mat[i])
        prior = pi[i]
        gauss_components[i] = likelihood * prior
    g = gauss_components / np.sum(gauss_components, axis=0)
    log_likelihood = np.sum(np.log(np.sum(gauss_components, axis=0)))
    return g, log_likelihood
