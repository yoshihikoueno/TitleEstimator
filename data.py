'''
data input/output module
'''
# built-in
import os
import argparse
import pdb
import json
import pickle
from multiprocessing import cpu_count

# external
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# customs
import utils

def load_doc_title(path, document_id_prefix='doc', apply_preprocess=True):
    '''
    load documents and titles

    Args:
        path: path to the data file
        document_id_prefix: prefix of document ids
        apply_preprocess: whether this func should apply preprocess
            this will ignored when path points to a pickle file

    Returns:
        tuple of dicts: documents, titles
        key: id, val: content
    '''
    if os.path.splitext(path)[-1] != '.pickle':
        data = pd.read_json(path)
        data.columns = 'id', 'content'

        if apply_preprocess:
            data.content = preprocess(data.content)
    else:
        with open(path, 'rb') as f:
            data = pickle.load(f)

    is_document = data.id.apply(lambda x: x[:len(document_id_prefix)] == document_id_prefix)
    documents = data[is_document]
    titles = data[~is_document]
    return documents, titles


def preprocess(series, npartitions=None):
    '''
    preprocess documents.
    1. normalize
    2. split to words

    Args:
        series: pd.Series of documents
        npartitions: the number of partitions to split
            None: auto

    Returns:
        series
    '''
    if npartitions is None:
        npartitions = cpu_count() * 4

    print('normalize')
    with ProgressBar():
        series = dd.from_pandas(series, npartitions=npartitions)\
            .apply(utils.normalize_string, meta=series)\
            .compute(scheduler='processes')

    print('split to words')
    with ProgressBar():
        series = dd.from_pandas(series, npartitions=npartitions)\
            .apply(utils.split2words, meta=('content', 'object'))\
            .compute(scheduler='processes')
    return series


def load_train(path):
    '''
    load train data
    '''
    data = pd.read_json(path)
    return data


def load_val(path):
    '''
    load val data
    '''
    data = pd.read_json(path)
    return data


def load_test(path):
    '''
    load test data
    '''
    data = pd.read_json(path)
    return data


def dump_prediction(prediction, path):
    '''
    save prediction result to a file

    Args:
        prediction: prediction results
        path: output path

    Returns:
        None
    '''
    if not isinstance(prediction, pd.DataFrame):
        raise NotImplementedError('Implement conversion to DF')

    prediction.to_json(path)
    return
