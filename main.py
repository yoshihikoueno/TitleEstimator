'''
Sorts title candidates for a given document
'''

# built-in
import os
import argparse
import pdb
import json
import pickle
import logging

# external
import pandas as pd

# customs
import data
import engine
import utils


logging.basicConfig(level=logging.DEBUG)

def main(
    data_path,
    train_data_path,
    val_data_path,
    test_data_path,
    output_path,
    prediction_name='suggestion.json',
    cache_dir=None,
):
    '''
    train a model and make a prediction

    Args:
        data_path: path to the data json file
        train_data_path: path to the train data
        val_data_path: path to the val data
        test_data_path: path to the test data
        output_path: path to the output dir
        prediction_name: the name of prediction output file
        cache_dir: where to save cache

    Returns:
        None
    '''
    # load data
    print('Loading data')
    documents, titles = data.load_doc_title(data_path)
    train_data = data.load_train(train_data_path)
    val_data = data.load_val(val_data_path)
    test_data = data.load_test(test_data_path)

    # convert to corpus
    print('Preparing corpus')
    dictionary = utils.make_dictionary(
        documents.content,
        cache_path=os.path.join(cache_dir, 'dictionary'),
    )
    documents['bow'] = utils.make_corpus(documents.content, dictionary)
    titles['bow'] = utils.make_corpus(titles.content, dictionary)

    # train
    print('Training model')
    model = engine.train(
        train_data,
        val_data,
        output_path,
        documents,
        titles,
        dictionary,
    )
    pdb.set_trace()

    # inference
    prediction = engine.predict(model, test_data, documents, titles)
    prediction_output = os.path.join(output_path, prediction_name)
    data.dump_prediction(prediction, prediction_output)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='data/exam_data1.json')
    parser.add_argument('--train_data_path', default='data/train_q.json')
    parser.add_argument('--val_data_path', default='data/val_q.json')
    parser.add_argument('--test_data_path', default='data/test_q.json')
    parser.add_argument('--output_path', default='./temp_output')
    parser.add_argument('--cache_dir', default='./data')

    args = parser.parse_args()
    main(**vars(args))
