'''
data input/output module
'''
# built-in
import os
import argparse
import pdb
import json
import pickle

# external
import pandas as pd
import tensorflow as tf

def load_doc_title(path, document_id_prefix='doc'):
    '''
    load documents and titles

    Args:
        path: path to the data file
        document_id_prefix: prefix of document ids

    Returns:
        tuple of dicts: documents, titles
        key: id, val: content
    '''
    data = pd.read_json()
    data.columns = 'id', 'content'
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
