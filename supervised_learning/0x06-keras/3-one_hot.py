#!/usr/bin/env python3
"""
module one_hot
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix
    Args:
        labels (vector): vector to be converted
        classes (int, optional): number of classes. Defaults to None.
    Returns:
        one-hot matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)
