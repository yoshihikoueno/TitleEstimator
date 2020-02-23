'''
provides various funcitons related to model
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
from tensorflow.keras import Model
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# customs
import utils


def train(train_data, val_data, output_path, documents, titles):
    '''
    train/val a model and save the trained model.

    Args:
        train_data (DataFrame): training data
        val_data (DataFrame): validation data
        output_path: where to save models
        documents: mapping of doc ID to doc content
        titiles: mapping of titles ID to title content

    Returns:
        model object
    '''
    model = Model()
    return model


def predict(model, data):
    '''
    make a prediction

    Args:
        model: model object
        data: input data
        documents: mapping of doc ID to doc content
        titiles: mapping of titles ID to title content

    Returns:
        prediction results
    '''
    prediction = model.predict(data)
    return prediction


