#!/usr/bin/env python3
"""Module word2vec_model."""
import gensim


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a gensim word2vec model.

    Args:
        sentences (list): sentences to be trained on
        size (int, optional): dimensionality of the embedding layer. Defaults
                              to 100.
        min_count (int, optional): minimum number of ocurrences of a word for
                                   use in training. Defaults to 5.
        window (int, optional): maximum distance between the current and
                                predicted word within a sentence. Defaults to
                                5.
        negative (int, optional): size of negative sampling. Defaults to 5.
        cbow (bool, optional): determines the training type. Defaults to True.
        iterations (int, optional): number of iterations to train over.
                                    Defaults to 5.
        seed (int, optional): seed for the random number generator. Defaults
                              to 0.
        workers (int, optional): number of worker threads to train the model.
                                 Defaults to 1.
    Returns:
        the trained model
    """
    model = gensim.models.Word2Vec(sentences, min_count=min_count,
                                   iter=iterations, size=size, window=window,
                                   negative=negative, seed=seed, sg=cbow,
                                   workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.iter)
    return model
