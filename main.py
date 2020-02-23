'''
Sorts title candidates for a given document
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

# customs
import data
import engine


def main(
    data_path,
    train_data_path,
    val_data_path,
    test_data_path,
    output_path,
    prediction_name='suggestion.json',
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

    Returns:
        None
    '''
    # load data
    documents, titles = data.load_doc_title(data_path)
    train_data = data.load_train(train_data_path)
    val_data = data.load_val(val_data_path)
    test_data = data.load_test(test_data_path)

    # train
    model = engine.train(train_data, val_data, output_path, documents, titles)

    # inference
    prediciton = engine.predict(model, test_data, documents, titles)
    prediciton_output = os.path.join(output_path, prediction_name)
    data.dump_prediction(prediciton, prediciton_output)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='data/exam_data1.json')
    parser.add_argument('--train_data_path', default='data/train_q.json')
    parser.add_argument('--val_data_path', default='data/val_q.json')
    parser.add_argument('--test_data_path', default='data/test_q.json')
    parser.add_argument('--output_path', default='./temp_output')

    args = parser.parse_args()
    main(**vars(args))
