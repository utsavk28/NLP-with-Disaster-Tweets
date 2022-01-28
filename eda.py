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
from sklearn.utils.class_weight import compute_class_weight
from config import INPUT_PATH, OUTPUT_PATH
from utils import check_target_dist
from collections import Counter

input_path = INPUT_PATH
output_path = OUTPUT_PATH
train = pd.read_csv(f'{input_path}/train.csv')
test = pd.read_csv(f'{input_path}/test.csv')
sample_submission = pd.read_csv(f'{input_path}/sample_submission.csv')
print(train.head())

X, y = train['text'], train['target']

print(train.shape)
print(check_target_dist(train['target']))
pd.DataFrame({'train': train.isna().sum(), 'test': test.isna().sum()})

cws = compute_class_weight('balanced', np.unique(y), y)
print(cws)

counter = Counter()
for ds in train:
    for text in ds:
        for word in text.split():
            counter[word] += 1

max_features = len(counter)
print(max_features)
counter.most_common(10)
