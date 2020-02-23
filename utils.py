'''
utility moduel
'''
# built-in
import os
import pdb
import json
import pickle
from multiprocessing import cpu_count

# external
import MeCab
import tensorflow as tf
import pandas as pd
import jaconv
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


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


def to_tagged_docs(docs):
    '''
    convert documents to genism tagged docs

    Args:
        docs (DataFrame): doc id and list of words in a doc
    '''
    with ProgressBar():
        tagged_docs = dd.from_pandas(docs, npartitions=cpu_count() * 4)\
            .apply(to_tagged_doc, axis=1)\
            .compute()
    return tagged_docs


def to_tagged_doc(doc):
    '''
    convert a single doc to genism TaggedDocument

    Args:
        doc (Series): should contain 'id', 'content'

    Returns:
        TaggedDocument
    '''
    tagged = TaggedDocument(words=doc.content, tags=[doc.id])
    return tagged
