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
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# customs
import utils

def load_doc_title(
    path,
    document_id_prefix='doc',
    apply_preprocess=True,
    cache_path=None,
):
    '''
    load documents and titles

    Args:
        path: path to the data file
        document_id_prefix: prefix of document ids
        apply_preprocess: whether this func should apply preprocess
            this will ignored when path points to a pickle file
        cache_path: where to save/load cache

    Returns:
        tuple of dicts: documents, titles
        key: id, val: content
    '''
    if cache_path is not None and os.path.exists(cache_path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = pd.read_json(path)
        data.columns = 'id', 'content'
        data = data.set_index('id')
        if apply_preprocess:
            data.content = preprocess(data.content)
        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)

    is_document = data.index.to_series().apply(
        lambda x: x[:len(document_id_prefix)] == document_id_prefix
    )
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

    print('preprocess')
    with ProgressBar():
        series = dd.from_pandas(series, npartitions=npartitions)\
            .apply(utils.normalize_string, meta=series)\
            .apply(utils.remove_white, meta=series)\
            .apply(utils.remove_symbol, meta=series)\
            .apply(utils.remove_digit, meta=series)\
            .apply(utils.split2words, meta=('content', 'object'))\
            .apply(utils.remove_too_short, meta=('content', 'object'))\
            .apply(utils.remove_stopwords, meta=('content', 'object'))\
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
