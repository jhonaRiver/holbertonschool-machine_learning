#!/usr/bin/env python3
"""Module tf_idf."""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Create a TF-IDF embedding.

    Args:
        sentences (list): sentences to analyze
        vocab (list, optional): vocabulary words to use for the analysis.
                                Defaults to None.
    Returns:
        embeddings: contains the embeddings
        features: features used for embeddings
    """
    if vocab is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names_out()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)
    embedding = X.toarray()
    return embedding, vocab
