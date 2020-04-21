from nltk.stem import PorterStemmer
from os import listdir


def stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in tokens]


def get_queries_output(directory='Queries/dev_output'):
    output = {}
    for sub_dir in listdir(directory):
        path = directory + '/' + sub_dir
        output_retrieved = load_document(path).split('\n')
        output[sub_dir[0]] = ["./pa1-data/" + s  for s in output_retrieved]
    return output


def load_document(filename):
    with open(filename) as f:
        return f.read().rstrip()