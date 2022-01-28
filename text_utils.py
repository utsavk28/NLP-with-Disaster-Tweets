import re
import string
import contractions
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def lower_case(text):
    return text.lower()


def accented_chars_to_ascii(text):
    text = unidecode.unidecode(text)
    return text


def url_pattern(text):
    return not (text.startswith('https://') or text.startswith('http://') or text.startswith('www.'))


def remove_urls(text):
    text = text.split()
    text = filter(url_pattern, text)
    return " ".join(text)


def handle_contractions(text):
    return contractions.fix(text)


def remove_digits(text):
    return re.sub(r'\w*\d\w*', ' ', text)


def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def drop_small_words(text, num=2):
    text = text.split()
    text = filter(lambda x: len(x) > num, text)
    return " ".join(text)
