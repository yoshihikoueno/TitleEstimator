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
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.callbacks import Callback
from gensim.corpora import Dictionary


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


def make_dictionary(docs, no_above=.5):
    '''
    make corpus

    Args:
        docs (series): a series that contains list of words
        no_above: words that appear in more than `no_above`% documents are filtered out

    Returns:
        dictionary
    '''
    dictionary = Dictionary(docs.tolist())
    dictionary.filter_extremes(no_above=no_above)
    dictionary[0]
    return dictionary


def make_corpus(docs, dictionary):
    '''
    make corpus

    Args:
        docs (series): a series that contains list of words
        dictionary: gensim.corpora.Dictionary object

    Returns:
        corpus
    '''
    corpus = list(map(dictionary.doc2bow, docs))
    return corpus


class EpochSaver(Callback):
    '''Callback to save model after each epoch.'''

    def __init__(self, path, prefix=None):
        '''
        Args:
            path: where to save models
            prefix: model file prefix
        '''
        self.path = path
        self.prefix = prefix
        self.epoch = 0
        self.logger = None

        if not os.path.exists(path):
            os.makedirs(path)

    def on_epoch_end(self, model):
        model.save(self._get_output_path())
        self.epoch += 1

    def _get_output_path(self):
        if self.prefix is not None:
            filename = f'{self.prefix}_epoch{self.epoch}.model'
        else:
            filename = f'epoch{self.epoch}.model'
        output_path = os.path.join(self.path, filename)
        return output_path


class EpochLogger(Callback):
    '''Callback to log information about training'''

    def __init__(self, log_start=False):
        self.log_start = log_start
        self.epoch = 0
        self.logger = None

    def on_epoch_begin(self, model):
        if self.log_start:
            print(f'Epoch {self.epoch}/{model.epochs} start')

    def on_epoch_end(self, model):
        print(f'Epoch {self.epoch}/{model.epochs} end')
        self.epoch += 1


class SupervisedEvalute(Callback):
    '''
    callback to evaluate model based on provided label data
    '''
    def __init__(self, val_data, docs, titles):
        self.val_data = val_data
        self.docs = docs
        self.titles = titles
        self.logger = None
        return

    def on_epoch_end(self, model):
        val_results = self.val_data.apply(
            sort_candidates,
            docs=self.docs,
            titles=self.titles,
            model=model,
            axis=1,
        )
        mrr = calculate_MRR(val_results)
        print(f'MRR: {mrr}')
        return


def calculate_MRR(val_data):
    '''calculate MRR'''
    data = val_data.apply(
        lambda series: 1 / series.candidates.index(series.ans_id),
        axis=1,
    )
    mrr = data.mean()
    return mrr


def sort_candidates(series, docs, titles, model):
    '''
    sort candidate titles contained in series.

    Args:
        series: pd.Series with index[title_id, candidates]
        docs: pd.DataFrame with columns[content, bow]
        titles: pd.DataFrame with columns[content, bow]
        model: gensim topic model

    Returns:
        series
    '''
    title_info = docs.loc[series.title_id]

    series.candidates = sorted(
        series.candidates,
        key=lambda doc_id: get_coherence(
            model[docs.loc[doc_id].bow],
            title_info.bow
        ),
    )
    return series


def get_coherence(topic, title):
    '''
    get coherence between topic and title

    Args:
        topic: list of (word_id, probability)
        titile: BoW of titile

    Returns:
        float: coherence
    '''
    topic = dict(topic)
    title = dict(title)

    coherence = sum(map(
        lambda word, freq: topic.get(word, 0) * freq,
        title.keys(), title.values(),
    ))
    return coherence
