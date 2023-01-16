# 0x10 Natural Language Processing - Evaluation Metrics

>

At the end of this project I was able to answer these conceptual questions:

* What are the applications of natural language processing?
* What is a BLEU score?
* What is a ROUGE score?
* What is perplexity?
* When should you use one evaluation metric over another?

## Tasks

0. Write the function `def uni_bleu(references, sentence):` that calculates the unigram BLEU score for a sentence:

    * `references` is a list of reference translations
        * each reference translation is a list of the words in the translation
    * `sentence` is a list containing the model proposed sentence
    * Returns: the unigram BLEU score

1. Write the function `def ngram_bleu(references, sentence, n):` that calculates the n-gram BLEU score for a sentence:

    * `references` is a list of reference translations
        * each reference translation is a list of the words in the translation
    * `sentence` is a list containing the model proposed sentence
    * `n` is the size of the n-gram to use for evaluation
    * Returns: the n-gram BLEU score

2. Write the function `def cumulative_bleu(references, sentence, n):` that calculates the cumulative n-gram BLEU score for a sentence:

    * `references` is a list of reference translations
        * each reference translation is a list of the words in the translation
    * `sentence` is a list containing the model proposed sentence
    * `n` is the size of the largest n-gram to use for evaluation
    * All n-gram scores should be weighted evenly
    * Returns: the cumulative n-gram BLEU score

## Results

| Filename |
| ------ |
| [0-uni_bleu.py](https://github.com/jhonaRiver/holbertonschool-machine_learning/blob/master/supervised_learning/0x10-nlp_metrics/0-uni_bleu.py)|
| [1-ngram_bleu.py](https://github.com/jhonaRiver/holbertonschool-machine_learning/blob/master/supervised_learning/0x10-nlp_metrics/1-ngram_bleu.py)|
| [2-cumulative_bleu.py](https://github.com/jhonaRiver/holbertonschool-machine_learning/blob/master/supervised_learning/0x10-nlp_metrics/2-cumulative_bleu.py)|
