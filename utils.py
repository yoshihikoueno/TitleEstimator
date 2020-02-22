'''
utility moduel
'''
# built-in
import os
import pdb
import json
import pickle

# external
import MeCab
import tensorflow as tf
import pandas as pd
import jaconv


def split2words(sentence):
    '''
    split a sentence to words

    Args:
        sentence: input sentence

    Returns:
        list of words
    '''
    tagger = MeCab.Tagger()
    tokens = tagger.parse(sentence).splitlines()[:-1]
    words = list(map(lambda x: x.split('\t')[0], tokens))
    return words


def normalize_string(string):
    '''
    normalize string.
    Half-width katakana -> Full-width katakana
    Full-width digit -> Half-width digit
    etc.

    Args:
        string: input string

    Returns:
        normalized string
    '''
    normalized = jaconv.normalize(string)
    return normalized
