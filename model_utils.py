import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, ReLU, Flatten
from keras.layers.merge import Concatenate
from keras import Model
from nltk.stem import WordNetLemmatizer
from keras.layers import Dropout, Dense, GRU, Embedding, LSTM, Bidirectional, Input,Activation
from keras.models import Sequential
from config import INPUT_PATH, OUTPUT_PATH, EARLY_STOPPING, VERBOSE, DROP_OUT, H_Node, EPOCHS, BATCH_SIZE, MAX_LEN, \
    EMBEDDING_DIM,MAX_POOL

nclasses = 2

dense_layers = [
    Dense(256, activation='relu'),
    Dropout(DROP_OUT),
    Dense(64, activation='relu'),
    Dropout(DROP_OUT),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
]


def time_series_layer(n_count, fun):
    layers = []
    for i in range(n_count - 1):
        layers.append(fun(H_Node, return_sequences=True))
        layers.append(Dropout(DROP_OUT))
    layers.append(fun(H_Node))
    layers.append(Dropout(DROP_OUT))
    return layers


def time_series_layer2(n_count, fun):
    layers = []
    for i in range(n_count - 1):
        layers.append(Bidirectional(fun(H_Node, return_sequences=True)))
        layers.append(Dropout(DROP_OUT))
    layers.append(Bidirectional(fun(H_Node)))
    layers.append(Dropout(DROP_OUT))
    return layers


def get_rnn_models(tokenizer, embedding_matrix):
    model1 = Sequential([
        Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                  input_length=MAX_LEN, trainable=True),
        *time_series_layer(2, LSTM),
        *dense_layers
    ])

    model2 = Sequential([
        *time_series_layer(2, LSTM),
        *dense_layers
    ])

    model3 = Sequential([
        Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                  input_length=MAX_LEN, trainable=True),
        *time_series_layer(2, GRU),
        *dense_layers
    ])

    model4 = Sequential([
        *time_series_layer(2, GRU),
        *dense_layers
    ])

    model5 = Sequential([
        *time_series_layer2(2, GRU),
        Dense(256, activation='relu'),
        Dropout(DROP_OUT),
        Dense(64, activation='relu'),
        Dropout(DROP_OUT),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model6 = Sequential([
        *time_series_layer2(2, LSTM),
        Dense(256, activation='relu'),
        Dropout(DROP_OUT),
        Dense(64, activation='relu'),
        Dropout(DROP_OUT),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model7 = Sequential([
        Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                  input_length=MAX_LEN, trainable=True),
        *time_series_layer2(2, GRU),
        Dense(256, activation='relu'),
        Dropout(DROP_OUT),
        Dense(64, activation='relu'),
        Dropout(DROP_OUT),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model8 = Sequential([
        Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                  input_length=MAX_LEN, trainable=True),
        *time_series_layer2(2, LSTM),
        Dense(256, activation='relu'),
        Dropout(DROP_OUT),
        Dense(64, activation='relu'),
        Dropout(DROP_OUT),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return [(model1, 0), (model2, 1), (model3, 0), (model4, 1),
            (model5, 1), (model6, 1), (model7, 0), (model8, 0)]


dense_layers = [
    Dense(256, activation='relu'),
    Dropout(DROP_OUT),
    Dense(64, activation='relu'),
    Dropout(DROP_OUT),
    Dense(16, activation='relu'),
    Dropout(DROP_OUT),
    Dense(2, activation='softmax')
]

layers1 = []
for i in range(1):
    layers1.append(Conv1D(H_Node, (i + 2)))
    layers1.append(BatchNormalization())
    layers1.append(ReLU())
    layers1.append(MaxPooling1D(MAX_POOL))
    layers1.append(Dropout(DROP_OUT))

layers2 = []
for i in [5, 30]:
    layers2.append(Conv1D(H_Node, 5))
    layers1.append(BatchNormalization())
    layers1.append(ReLU())
    layers1.append(Dropout(DROP_OUT))
    layers1.append(MaxPooling1D(i))


def get_cnn_models(tokenizer, embedding_matrix):
    model = Sequential()
    convs = []
    filter_sizes = []
    layer = 5
    for fl in range(0, layer):
        filter_sizes.append((fl + 2))

    embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_LEN,
                                trainable=True)
    sequence_input = Input(shape=(MAX_LEN,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(H_Node, kernel_size=fsz,
                        activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)
    l_cov1 = Conv1D(H_Node, 5, activation='relu')(l_merge)
    l_cov1 = Dropout(DROP_OUT)(l_cov1)
    l_pool1 = MaxPooling1D(5)(l_cov1)

    l_flat = Flatten()(l_pool1)
    l_dense = Dense(1024, activation='relu')(l_flat)
    l_dense = Dropout(DROP_OUT)(l_dense)
    l_dense = Dense(512, activation='relu')(l_dense)
    l_dense = Dropout(DROP_OUT)(l_dense)
    preds = Dense(2, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)
    return model


def get_rcnn_models(tokenizer, embedding_matrix):
    kernel_size = 2
    filters = 256
    pool_size = 2
    gru_node = 256
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_LEN,
                        trainable=True))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(gru_node, recurrent_dropout=0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))

    return model
