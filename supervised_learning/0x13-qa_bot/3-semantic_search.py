#!/usr/bin/env python3
"""Module semantic_search."""


def semantic_search(corpus_path, sentence):
    """
    Perform semantic search on a corpus of documents.

    Args:
        corpus_path (str): path to the corpus of reference documents on which
                           to perform semantic search
        sentence (str): sentence from which to perform semantic search
    Returns:
        reference text of the document most similar to sentence
    """
