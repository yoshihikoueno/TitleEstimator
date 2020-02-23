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

def load_doc_title(path, document_id_prefix='doc', preprocess=True):
    '''
    load documents and titles

    Args:
        path: path to the data file
        document_id_prefix: prefix of document ids
        preprocess: whether this func should apply preprocess

    Returns:
        tuple of dicts: documents, titles
        key: id, val: content
    '''
    data = pd.read_json(path)
    data.columns = 'id', 'content'

    tqdm.pandas(desc='normalize')
    data.content = data.content.progress_apply(utils.normalize_string)

    with ProgressBar():
        data.content = dd.from_pandas(data.content, npartitions=cpu_count() * 8)\
            .apply(utils.split2words, meta=('content', 'object'))\
            .compute(scheduler='processes')
    pdb.set_trace()

    is_document = data[0].apply(lambda x: x[len(document_id_prefix)] == document_id_prefix)
    documents = data[is_document]
    titles = data[~is_document]
    return documents, titles


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
