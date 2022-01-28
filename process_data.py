import pandas as pd
from sklearn.model_selection import train_test_split
from config import INPUT_PATH, OUTPUT_PATH, MAX_FEATURES
from utils import clean_text, tokenize_and_pad, tfidf_and_decomposition, load_glove, to_glove
import joblib

input_path = INPUT_PATH
output_path = OUTPUT_PATH
train = pd.read_csv(f'{input_path}/train.csv')
test = pd.read_csv(f'{input_path}/test.csv')

X, y = train['text'], train['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=613, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

X_train = clean_text(X_train)
X_val = clean_text(X_val)
X_test = clean_text(X_test)
test = clean_text(test['text'])

X_train_tf, X_val_tf, X_test_tf, test_tf = tfidf_and_decomposition(
    X_train, X_val, X_test, test)
data, tokenizer = tokenize_and_pad(train=X_train, max_features=MAX_FEATURES,
                                   data_arr=[X_train, X_val, X_test, test])
X_train_tk, X_val_tk, X_test_tk, test_tk = data

embedding_matrix = load_glove(tokenizer.word_index)
X_train_glove = to_glove(embedding_matrix, X_train_tk)
X_val_glove = to_glove(embedding_matrix, X_val_tk)
X_test_glove = to_glove(embedding_matrix, X_test_tk)
test_glove = to_glove(embedding_matrix, test_tk)


np.save(f'{OUTPUT_PATH}/X_train_clean_tf.npy',X_train_tf)
np.save(f'{OUTPUT_PATH}/X_val_clean_tf.npy',X_val_tf)
np.save(f'{OUTPUT_PATH}/X_test_clean_tf.npy',X_test_tf)
np.save(f'{OUTPUT_PATH}/test_clean_tf.npy',test_tf)

np.save(f'{OUTPUT_PATH}/X_train_clean_tk.npy',X_train_tk)
np.save(f'{OUTPUT_PATH}/X_val_clean_tk.npy',X_val_tk)
np.savev(f'{OUTPUT_PATH}/X_test_clean_tk.npy',X_test_tk)
np.save(f'{OUTPUT_PATH}/test_clean_tk.npy',test_tk)

np.save(f'{OUTPUT_PATH}/X_train_clean_glove.npy',X_train_glove)
np.save(f'{OUTPUT_PATH}/X_val_clean_glove.npy',X_val_glove)
np.save(f'{OUTPUT_PATH}/X_test_clean_glove.npy',X_test_glove)
np.save(f'{OUTPUT_PATH}/test_clean_glove.npy',test_glove)

np.save(f'{OUTPUT_PATH}/y_train.npy',y_train)
np.save(f'{OUTPUT_PATH}/y_val.npy',y_val)
np.save(f'{OUTPUT_PATH}/y_test.npy',y_test)

np.save(f'{OUTPUT_PATH}/embedding_matrix.npy',embedding_matrix)
joblib.dump(tokenizer, f'{OUTPUT_PATH}/model/tokenizer.pkl')
