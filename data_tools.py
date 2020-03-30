from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os


def load_sentence(path):
    f = open(path)
    content = f.readline()
    number_regexp = re.compile("^[0-9]*[,.]*[0-9]*$")
    stop_words = stopwords.words('english')
    tokenized = word_tokenize(content)
    filtered_sentence = [w for w in list(tokenized) if not w in stop_words]
    filtered_sentence = [w for w in filtered_sentence if len(w) > 1]
    filtered_sentence = [w for w in filtered_sentence if not number_regexp.match(w)]
    lemmatizer = WordNetLemmatizer()
    filtered_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence]
    return filtered_sentence


def create_index(root_dir):
    collection = set()
    for sub_dir in os.listdir(root_dir)[:2]:
        path = root_dir + '/' + sub_dir
        for filename in os.listdir(path):
            for term in get_terms('./' + path + '/' + filename):
                collection.add(term)
    return collection


def get_terms(filename):
    f = open(filename, 'r')
    content = f.readline().rstrip().split(' ')
    return content

