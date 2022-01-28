import os
import re
import string
import joblib
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
import contractions
import spacy
import unidecode
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from config import INPUT_PATH, OUTPUT_PATH
from models import ann_tfidf
from utils import plot_history, rnn_model, cnn_model, rcnn_model, bert_model

experiments = [ann_tfidf, rnn_model, cnn_model, rcnn_model, bert_model]

embedding_matrix = np.load(f'{OUTPUT_PATH}/embedding_matrix.npy')
tokenizer = joblib.load(f'{OUTPUT_PATH}/model/tokenizer.pkl')


def run_experiment(X_train, X_val, X_test, y_train, y_val, y_test, experiment):
    model, history = experiment(X_train, y_train, X_val, y_val)
    model.evaluate(X_train, y_train)
    model.evaluate(X_val, y_val)
    model.evaluate(X_test, y_test)

    plot_history(history)
    model.summary()
    print(classification_report(y_test, np.argmax(model.predict(X_test), axis=1)))


def run_experiment_seq(X_train, X_val, X_test, y_train, y_val, y_test, experiment, model_no):
    model, history = experiment(X_train, y_train, X_val, y_val, tokenizer, embedding_matrix, model_no)
    model.evaluate(X_train, y_train)
    model.evaluate(X_val, y_val)
    model.evaluate(X_test, y_test)
    plot_history(history)
    model.summary()
    print(classification_report(y_test, np.argmax(model.predict(X_test), axis=1)))


if __name__ == "__main__":
    experiment = experiments[0]
    model_no = 0
    X_train = np.load(f'{OUTPUT_PATH}/X_train_clean_glove.npy')
    X_val = np.load(f'{OUTPUT_PATH}/X_val_clean_glove.npy')
    X_test = np.load(f'{OUTPUT_PATH}/X_test_clean_glove.npy')
    test = np.load(f'{OUTPUT_PATH}/test_clean_glove.npy')

    y_train = np.load(f'{OUTPUT_PATH}/y_train.npy')
    y_val = np.load(f'{OUTPUT_PATH}/y_val.npy')
    y_test = np.load(f'{OUTPUT_PATH}/y_test.npy')
    run_experiment(X_train, X_val, X_test, y_train, y_val, y_test, experiment)
    run_experiment(X_train, X_val, X_test, y_train, y_val, y_test, experiment, model_no)
