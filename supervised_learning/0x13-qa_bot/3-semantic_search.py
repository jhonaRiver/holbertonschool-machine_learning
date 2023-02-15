#!/usr/bin/env python3
"""Module semantic_search."""
import numpy as np
import os
import tensorflow_hub as hub


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
    documents = [sentence]
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md') is False:
            continue
        with open(corpus_path + '/' + filename, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")
    embeddings = model(documents)
    correlation = np.inner(embeddings, embeddings)
    closest = np.argmax(correlation[0, 1:])
    similar = documents[closest + 1]
    return similar
