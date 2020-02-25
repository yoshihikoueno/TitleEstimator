'''
utility moduel
'''
# built-in
import os
import pdb
import json
import pickle
from multiprocessing import cpu_count
import unicodedata
import re

# external
import MeCab
import pandas as pd
import jaconv
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.callbacks import Callback
from gensim.corpora import Dictionary


def split2words(sentence, lemmatize=True):
    '''
    split a sentence to words

    Args:
        sentence: input sentence
        lemmatize: whether this func should lemmatize words

    Returns:
        list of words
    '''
    tagger = MeCab.Tagger()
    tokens = tagger.parse(sentence).splitlines()[:-1]
    if lemmatize:
        try:
            words = list(map(lambda x: x.split('\t')[1].split(',')[6], tokens))
        except:
            print(list(filter(lambda token: len(token.split('\t')) == 1, tokens)))
            print(f'last: {tokens[-1]}')
            for token in tokens:
                print()
                print(token)
                print(token.split('\t')[1])
                print(token.split('\t')[1].split(','))
            raise
    else:
        words = list(map(lambda x: x.split('\t')[0], tokens))
    return words


def remove_white(string):
    '''
    remove white spaces
    '''
    string = re.sub(r'\s', '', string)
    return string


def remove_digit(string):
    '''
    remove digits
    '''
    string = re.sub(r'\d', '', string)
    return string


def is_symbol(x):
    '''
    return if a char is symbols
    '''
    return unicodedata.category(x)[0] in 'PSZ'


def remove_symbol(string):
    '''
    remove symbol from string
    '''
    string = ''.join(filter(lambda x: not is_symbol(x), string))
    return string


def remove_too_short(words):
    '''
    remove too short words
    '''
    words = list(filter(
        lambda x: not re.match(r'^[あ-ん]{1,2}$', x),
        words,
    ))
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


def remove_stopwords(words):
    '''
    remove stop words
    '''
    stop_words = {
        'は', 'の', 'に', 'を', 'て', 'さ', 'から', 'も', 'として', 'な', 'や', 'など',
        'まで', 'へ', 'という', 'により', 'による', 'によって', 'か', 'において', 'について', 'のみ',
        'における', 'だけ', 'にて', 'とともに', 'ながら', 'に対して', 'と共に', 'ものの', 'にかけて',
        'たり', 'ほど', 'ので', 'といった', 'に関する', 'に', '対する', 'に対し', 'ん', 'しか',
        'にとって', 'つつ', 'に関して', 'わ', 'なさ', 'を通じて', 'よ', 'ずつ', 'ばかり', 'にわたって',
        'にあたる', 'ね', 'にも', 'こそ', 'を通して', 'かい', 'に際して', 'のに', 'をもって', 'さえ',
        'にわたり', 'すら', 'に従って', 'にあたって', 'って', 'にわたる', 'にあたり', 'に従い', 'べ',
        'ぜ', 'ぞ', 'ど', 'け', 'か所', 'にし', 'につき', 'ねん', 'に当たる', 'に際し', 'につれて', 'とか',
        'だり', 'につれ', 'をめぐって', 'てん', 'もん', 'に当たって', 'にまつわる', 'の子', 'にあ',
        'は元', 'を以て', 'デ', 'ぐらい', 'にかけ', 'やら', 'かな', 'しも', 'なんて', 'に関し',
    }
    words = list(filter(lambda w: w not in stop_words, words))
    return words


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
    return tagged_docs.tolist()


def to_tagged_doc(doc):
    '''
    convert a single doc to genism TaggedDocument

    Args:
        doc (Series): should contain 'id', 'content'

    Returns:
        TaggedDocument
    '''
    tagged = TaggedDocument(words=doc.content, tags=[doc.name])
    return tagged


def make_dictionary(docs, no_above=.5, cache_path=None, filter_=True):
    '''
    make corpus

    Args:
        docs (series): a series that contains list of words
        no_above: words that appear in more than `no_above`% documents are filtered out
        cache_path: path to the dictionary file

    Returns:
        dictionary
    '''
    if cache_path is not None and os.path.exists(cache_path):
        print('found dictionary cache. loading...')
        dictionary = Dictionary.load(cache_path)
    else:
        dictionary = Dictionary(docs.tolist())
        if filter_: dictionary.filter_extremes(no_above=no_above)

    dictionary[0]
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        dictionary.save(cache_path)
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


class SampleEvaluator(CallbackAny2Vec):
    '''Callback to evaluate a model after each epoch.'''

    def __init__(self, val_data, nsample, parent_model):
        '''
        Args:
            val_data: validation data
        '''
        self.val_data = val_data
        self.epoch = 0
        self.nsample = nsample
        self.parent_model = parent_model
        self.logger = None

    def on_epoch_end(self, model):
        model = self.parent_model.replica(model)
        results = model.validate(self.val_data.sample(self.nsample), concurrent=False)
        print(f'SampleValidation {results}  - Epoch {self.epoch}')
        self.epoch += 1


class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, path, interval=None):
        '''
        Args:
            path: where to save models
            interval: save interval
                None: disable this callback
        '''
        self.dirname = os.path.dirname(path)
        self.path = path
        self.epoch = 0
        self.interval = interval
        self.logger = None

        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

    def on_epoch_end(self, model):
        if (self.interval is not None) and (self.epoch % self.interval == 0):
            save_path = f'{self.path}_{self.epoch}'
            model.save(save_path)
            print(f'AutoSaved model to {save_path} - Epoch {self.epoch}')
        self.epoch += 1


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


def calculate_MRR(val_data):
    '''calculate MRR'''
    data = val_data.apply(
        lambda series: 1 / (series.candidates.index(series.ans_id) + 1),
        axis=1,
    )
    mrr = data.mean()
    return mrr


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
