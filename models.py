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
import contractions
import spacy
import unidecode
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ann_tfidf(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(512, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(256, activation='relu'),
        #     Dense(128,activation='relu'),
        Dense(64, activation='relu'),
        #     Dense(32,activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=10, batch_size=256, verbose=2)
