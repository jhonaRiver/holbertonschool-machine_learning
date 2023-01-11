# 0x0F Natural Language Processing - Word Embeddings

> In natural language processing (NLP), a **word embedding** is a representation of a word. The embedding is used in text analysis. Typically the representation is a real-valued vector that encodes the meaning of the word in such a way that words that are closer in the vector space are expected to be similar in meaning. Word embeddings can be obtained using language modeling and feature learning techniques where words or phrases from the vocabulary are mapped to vectors of real numbers.
> Methods to generate this mapping include neural networks, dimensionality reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge base method, and explicit representation in terms of the context in which words appear.
> Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing and sentiment analysis.

At the end of this project I was able to answer these conceptual questions:

* What is natural language processing?
* What is a word embedding?
* What is bag of words?
* What is TF-IDF?
* What is CBOW?
* What is a skip-gram?
* What is an n-gram?
* What is negative sampling?
* What is word2vec, GloVe, fastText, ELMo?

## Tasks

0. Write a function `def bag_of_words(sentences, vocab=None):` that creates a bag of words embedding matrix:

    * `sentences` is a list of sentences to analyze
    * `vocab` is a list of the vocabulary words to use for the analysis
        * If `None`, all words within `sentences` should be used
    * Returns: `embeddings, features`
        * `embeddings` is a `numpy.ndarray` of shape `(s, f)` containing the embeddings
            * `s` is the number of sentences in `sentences`
            * `f` is the number of features analyzed
        * `features` is a list of the features used for `embeddings`

1. Write a function `def tf_idf(sentences, vocab=None):` that creates a TF-IDF embedding:

    * `sentences` is a list of sentences to analyze
    * `vocab` is a list of the vocabulary words to use for the analysis
        * If `None`, all words within `sentences` should be used
    * Returns: `embeddings, features`
        * `embeddings` is a `numpy.ndarray` of shape `(s, f)` containing the embeddings
            * `s` is the number of sentences in `sentences`
            * `f` is the number of features analyzed
        * `features` is a list of the features used for `embeddings`

2. Write a function `def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):` that creates and trains a `gensim` `word2vec` model:

    * `sentences` is a list of sentences to be trained on
    * `size` is the dimensionality of the embedding layer
    * `min_count` is the minimum number of occurrences of a word for use in training
    * `window` is the maximum distance between the current and predicted word within a sentence
    * `negative` is the size of negative sampling
    * `cbow` is a boolean to determine the training type; `True` is for CBOW; `False` is for Skip-gram
    * `iterations` is the number of iterations to train over
    * `seed` is the seed for the random number generator
    * `workers` is the number of worker threads to train the model
    * Returns: the trained model

3. Write a function `def gensim_to_keras(model):` that converts a `gensim` `word2vec` model to a `keras` Embedding layer:

    * `model` is a trained `gensim` `word2vec` models
    * Returns: the trainable `keras` Embedding

4. Write a function `def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1):` that creates and trains a `genism` `fastText` model:

    * `sentences` is a list of sentences to be trained on
    * `size` is the dimensionality of the embedding layer
    * `min_count` is the minimum number of occurrences of a word for use in training
    * `window` is the maximum distance between the current and predicted word within a sentence
    * `negative` is the size of negative sampling
    * `cbow` is a boolean to determine the training type; `True` is for CBOW; `False` is for Skip-gram
    * `iterations` is the number of iterations to train over
    * `seed` is the seed for the random number generator
    * `workers` is the number of worker threads to train the model
    * Returns: the trained model

5. When training an ELMo embedding model, you are training:

    1. The internal weights of the BiLSTM
    2. The character embedding layer
    3. The weights applied to the hidden states

    In the text file 5-elmo, write the letter answer, followed by a newline, that lists the correct statements:

    * A. 1, 2, 3
    * B. 1, 2
    * C. 2, 3
    * D. 1, 3
    * E. 1
    * F. 2
    * G. 3
    * H. None of the above

## Results

| Filename |
| --- |
| [0-bag_of_words.py]()|
| [1-tf_idf.py]()|
| [2-word2vec.py]()|
| [3-gensim_to_keras.py]()|
| [4-fasttext.py]()|
| [5-elmo]()|
