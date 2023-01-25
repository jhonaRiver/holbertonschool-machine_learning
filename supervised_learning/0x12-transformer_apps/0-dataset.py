#!/usr/bin/env python3
"""Module Dataset class."""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Load and prep a dataset for machine translation."""

    def __init__(self):
        """Class constructor."""
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True, as_supervised=True)
        self.data_train, self.data_valid = examples['train'], \
            examples['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Create sub-word tokenizers for our dataset.

        Args:
            data (Dataset): examples formatted
        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=(2 ** 15))
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=(2 ** 15))
        return tokenizer_pt, tokenizer_en
