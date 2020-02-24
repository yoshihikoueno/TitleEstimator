'''
provides various funcitons related to model
'''
# built-in
import os
import argparse
import pdb
import json
import pickle
from multiprocessing import cpu_count

# external
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import LdaModel
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.callbacks import Callback
from gensim.models.callbacks import CoherenceMetric
from gensim.models.callbacks import ConvergenceMetric

# customs
import utils


def train(
    train_data,
    val_data,
    output_path,
    documents,
    titles,
    dictionary,
    num_topics=1000,
    iterations=400,
    chunksize=2000,
    passes=20,
    eval_every=1,
):
    '''
    train/val a model and save the trained model.

    Args:
        train_data (DataFrame): training data
        val_data (DataFrame): validation data
        output_path: where to save models
        documents: mapping of doc ID to doc content
        titiles: mapping of titles ID to title content
        dictionary: gensim.Dictionary object
        num_topics: the number of topics
        iterations: train iterations
        eval_every: eval model every `eval_every` iterations

    Returns:
        model object
    '''
    val_data = pd.concat([train_data, val_data], ignore_index=True)
    model = LdaModel(
        corpus=documents.bow.tolist(),
        id2word=dictionary.id2token,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        chunksize=chunksize,
        eval_every=eval_every,
        callbacks=[
            # utils.EpochSaver(output_path),
            # utils.EpochLogger(log_start=True),
            # utils.SupervisedEvalute(val_data, documents, titles)
            # CoherenceMetric(corpus=documents.bow, logger='shell'),
            # ConvergenceMetric(logger='shell'),
        ],
    )
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


