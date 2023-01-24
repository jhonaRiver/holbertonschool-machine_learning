#!/usr/bin/env python3
"""Module Dataset class."""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Load and prep a dataset for machine translation."""

    def __init__(self):
        """Class constructor."""

    def tokenize_dataset(self, data):
        """
        Create sub-word tokenizers for our dataset.

        Args:
            data (Dataset): examples formatted
        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """

    def encode(self, pt, en):
        """
        Encode a translation into tokens.

        Args:
            pt (Tensor): contains the Portuguese sentence
            en (Tensor): contains the corresponding English sentence
        Returns:
            pt_tokens: contains the Portuguese tokens
            en_tokens: contains the English tokens
        """

    def tf_encode(self, pt, en):
        """
        Act as a wrapper for encode.

        Args:
            pt (Tensor): contains the Portuguese sentence
            en (Tensor): contains the corresponding English sentence
        """
