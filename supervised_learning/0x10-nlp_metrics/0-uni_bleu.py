#!/usr/bin/env python3
"""Module uni_bleu."""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score for a sentence.

    Args:
        references (list): reference translations
        sentence (list): contains the model propodes sentence
    Returns:
        unigram BLEU score
    """
    sen = list(set(sentence))
    count_dict = {}

    for reference in references:
        for word in reference:
            if word in sen:
                if word not in count_dict.keys():
                    count_dict[word] = reference.count(word)
                else:
                    new = reference.count(word)
                    old = count_dict[word]
                    count_dict[word] = max(new, old)
    len_sen = len(sentence)
    list_references = []
    for reference in references:
        len_ref = len(reference)
        list_references.append(((abs(len_ref - len_sen)), len_ref))
    reference_len = sorted(list_references, key=lambda x: x[0])
    reference_len = reference_len[0][1]
    if len_sen > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - (float(reference_len) / len_sen))
    bleu_score = bp * np.exp(np.log(sum(count_dict.values()) / len_sen))
    return bleu_score
