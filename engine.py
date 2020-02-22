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


def train(train_data, val_data, output_path):
    '''
    train/val a model and save the trained model.

    Args:
        train_data (DataFrame): training data
        val_data (DataFrame): validation data
        output_path: where to save models

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

    Returns:
        prediction results
    '''
    prediction = model.predict(data)
    return prediction


