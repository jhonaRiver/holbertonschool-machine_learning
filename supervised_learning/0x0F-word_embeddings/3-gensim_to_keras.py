#!/usr/bin/env python3
"""Module gensim_to_keras."""


def gensim_to_keras(model):
    """
    Convert a gensim word2vec model to a keras Embedding layer.

    Args:
        model (gensim): trained models
    Returns:
        trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=False)
