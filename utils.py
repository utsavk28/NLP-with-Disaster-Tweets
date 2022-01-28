import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from config import MAX_LEN
from config import INPUT_PATH, EMBEDDING_DIST_MEAN, EMBEDDING_DIST_STD
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from text_utils import lower_case, accented_chars_to_ascii, remove_urls, handle_contractions, \
    remove_digits, remove_punctuations, remove_stopwords, lemmatize_words, drop_small_words


def clean_text(x):
    x = x.fillna('')
    x = x.apply(lower_case)
    x = x.apply(accented_chars_to_ascii)
    #     x = x.apply(remove_urls)
    x = x.apply(handle_contractions)
    #     x = x.apply(remove_digits)
    x = x.apply(remove_punctuations)
    x = x.apply(remove_stopwords)
    x = x.apply(lemmatize_words)
    #     x = x.apply(drop_small_words)
    x = x.fillna('UNK')
    return x


def check_target_dist(data):
    return pd.DataFrame(data.value_counts())


def tokenize_and_pad(train, max_features, data_arr):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train)
    tokenized_data = []
    for data in data_arr:
        data = pad_sequences(tokenizer.texts_to_sequences(data), maxlen=MAX_LEN)
        tokenized_data.append(data)
    return tokenized_data,tokenizer


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')[:300]


def load_glove(word_index, max_features, ):
    EMBEDDING_FILE = f'{INPUT_PATH}/glove840b300dtxt/glove.840B.300d.txt'
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_ems = np.stack(embeddings_index.values())
    emb_mean, emb_std = EMBEDDING_DIST_MEAN, EMBEDDING_DIST_STD
    embed_size = all_ems.shape[1]

    nb_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_glove(matrix,data) :
    return np.array([matrix[i] for i in data])

def tfidf_and_decomposition(X_train, X_val, X_test, test):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    tfidf.fit(pd.concat([X_train, X_val]))
    X_train_tfidf = tfidf.transform(X_train).todense()
    X_val_tfidf = tfidf.transform(X_val).todense()
    X_test_tfidf = tfidf.transform(X_test).todense()
    test_tfidf = tfidf.transform(test).todense()

    svd = TruncatedSVD(n_components=25, random_state=42)
    svd.fit(X_train_tfidf)
    X_train_tfidf = svd.transform(X_train_tfidf)
    X_val_tfidf = svd.transform(X_val_tfidf)
    X_test_tfidf = svd.transform(X_test_tfidf)
    test_tfidf = svd.transform(test_tfidf)
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, test_tfidf


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
