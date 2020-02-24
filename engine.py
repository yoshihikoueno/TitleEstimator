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

def train_lstm(
    train_data,
    val_data,
    output_path,
    documents,
    titles,
    dictionary,
):
    '''
    train lstm
    '''
    return model


class CustomLDA:
    '''
    custimized lda model
    '''
    def __init__(self, documents, titles, dictionary,):
        self.documents = documents
        self.titles = titles
        self.dictionary = dictionary
        self.model = None
        return

    def train(
        self,
        train_data,
        val_data,
        output_path,
        num_topics=1000,
        iterations=100,
        chunksize=2000,
        passes=1,
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
        self.model = LdaModel(
            corpus=self.documents.bow.tolist(),
            id2word=self.dictionary.id2token,
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
        return self

    def validate(self, data):
        '''validate this model'''
        prediction = self.predict(data)
        mrr = utils.calculate_MRR(prediction)
        return mrr

    def predict(self, data):
        '''
        make a prediction

        Args:
            data: input data

        Returns:
            prediction results
        '''
        data = data.apply(
            self.sort_candidates,
            axis=1,
        )
        return data

    def sort_candidates(self, series, log_before=False, log_after=False):
        '''
        sort candidate titles contained in series.

        Args:
            series: pd.Series with index[title_id, candidates]
            docs: pd.DataFrame with columns[content, bow]
            titles: pd.DataFrame with columns[content, bow]
            model: gensim topic model
            log_before: whether this func should log candidates before sorting
            log_after: whether this func should log candidates after sorting

        Returns:
            series
        '''
        title_info = self.titles.loc[series.title_id]

        if log_before:
            print(list(map(
                lambda doc_id: utils.get_coherence(
                    self.model.get_document_topics(self.documents.loc[doc_id].bow),
                    title_info.bow
                ),
                series.candidates,
            )))

        series.candidates = sorted(
            series.candidates,
            key=lambda doc_id: - utils.get_coherence(
                self.model.get_document_topics(self.documents.loc[doc_id].bow, 0),
                title_info.bow
            ),
        )

        if log_after:
            print(list(map(
                lambda doc_id: self.get_coherence(
                    self.model.get_document_topics(self.documents.loc[doc_id].bow, 0),
                    title_info.bow
                ),
                series.candidates,
            )))
            print()
        return series
