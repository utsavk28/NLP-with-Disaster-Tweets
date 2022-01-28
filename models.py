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
from tensorflow.keras.layers import BatchNormalization, ReLU
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, ReLU, Flatten
from keras.layers.merge import Concatenate
from keras import Model
from nltk.stem import WordNetLemmatizer
from keras.layers import Dropout, Dense, GRU, Embedding, LSTM, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from config import INPUT_PATH, OUTPUT_PATH, EARLY_STOPPING, VERBOSE, DROP_OUT, H_Node, EPOCHS, BATCH_SIZE, MAX_LEN
from model_utils import get_cnn_models, get_rnn_models, get_rcnn_models
import tensorflow_hub as hub
import tensorflow_text as text

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=EARLY_STOPPING)
metrics = ['accuracy']


# ANN TFIDF
def ann_tfidf(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(512, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=10, batch_size=256, verbose=2)
    return model, history


# RNN,LSTM,GRU,Bidirectional
def rnn_model(X_train, y_train, X_val, y_val, tokenizer, embedding_matrix, model_no):
    model = get_rnn_models(tokenizer, embedding_matrix)[model_no][0]
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=metrics)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[callback])
    return model, history


# CNN
def cnn_model(X_train, y_train, X_val, y_val, tokenizer, embedding_matrix, model_no):
    model = get_cnn_models(tokenizer, embedding_matrix)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=metrics)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[callback])
    return model, history


def rcnn_model(X_train, y_train, X_val, y_val, tokenizer, embedding_matrix, model_no):
    model = get_rcnn_models(tokenizer, embedding_matrix)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=metrics)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[callback])
    return model, history


def bert_model(X_train, y_train, X_val, y_val):
    preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text-layer')
    preprocessed_text = preprocess(text_input)
    outputs = encoder(preprocessed_text)
    d_layer = tf.keras.layers.Dropout(0.1, name="dropout-layer")(outputs['pooled_output'])
    d_layer = tf.keras.layers.Dense(2, activation='softmax', name="output")(d_layer)
    model = tf.keras.Model(inputs=[text_input], outputs=[d_layer])
    model.compile(optimizer=Adam(3e-4), loss='sparse_categorical_crossentropy', metrics=metrics)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[callback])
    return model, history
