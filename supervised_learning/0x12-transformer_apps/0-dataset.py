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
