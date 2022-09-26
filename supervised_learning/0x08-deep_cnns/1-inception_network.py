#!/usr/bin/env python3
"""
module inception_network
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network as described in Going Deeper with
    Convolutions(2014)
    Returns:
        keras model
    """
