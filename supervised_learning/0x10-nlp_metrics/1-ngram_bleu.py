#!/usr/bin/env python3
"""Module ngram_bleu."""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculate the n-gram BLEU score for a sentence.

    Args:
        references (list): reference translations
        sentence (list): contains the model proposed sentence
        n (int): size of the n-gram to use for evaluation
    Returns:
        n-gram BLEU score
    """
    count_dict = {}
    c_grams = ngram(sentence, n)
    c_grams = list(set(c_grams))
    len_trans = len(c_grams)
    ref_grams = []
    for reference in references:
        list_grams = ngram(reference, n)
        ref_grams.append(list_grams)
    for grams in ref_grams:
        for word in grams:
            if word in c_grams:
                if word not in count_dict.keys():
                    count_dict[word] = grams.count(word)
                else:
                    curr = grams.count(word)
                    prev = count_dict[word]
                    count_dict[word] = max(curr, prev)
    precision = sum(count_dict.values()) / len_trans
    best_match_lst = []
    for reference in references:
        ref_len = len(reference)
        diff = abs(ref_len - len(sentence))
        best_match_lst.append((diff, ref_len))
    arranged_lst = sorted(best_match_lst, key=lambda x: x[0])
    best_match = arranged_lst[0][1]
    if len(sentence) > best_match:
        bp = 1
    else:
        bp = np.exp(1 - (float(best_match) / len(sentence)))
    Bleu_score = bp * np.exp(np.log(precision))
    return Bleu_score


def ngram(sentence, n):
    """
    Create gram.

    Args:
        sentence (list): contains the model proposed sentence
        n (int): size of the n-gram to use for evaluation
    Returns:
        list of new grams
    """
    lst_gram = []
    for i in range(len(sentence)):
        first = i + n
        last = i
        if first >= len(sentence) + 1:
            break
        aux = sentence[last: first]
        result = ' '.join(aux)
        lst_gram.append(result)
    return lst_gram
