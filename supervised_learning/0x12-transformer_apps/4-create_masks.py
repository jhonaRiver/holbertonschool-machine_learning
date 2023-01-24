#!/usr/bin/env python3
"""Module create_masks."""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def create_masks(inputs, target):
    """
    Create all masks for training/validation.

    Args:
        inputs (Tensor): contains the input sentence
        target (Tensor): contains the target sentence
    Returns:
        encoder_mask: padding mask to be applied
        combined_mask: padding mask used in the 1st attention block
        decoder_mask: padding mask used in the 2nd attention block
    """
