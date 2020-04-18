from utils import stem
from operator import itemgetter

def create_vocabulary(collection):
    vocabulary = dict()
    for tokens in collection.values()
        for token in tokens:
            if not token in vocabulary:
                vocabulary[token] = 0
            vocabulary[token] += 1
    return vocabulary

def create_language_model(collection, vocabulary, mu):
    index = dict()
    for document in collection:
        index[document] = dict()
        n = len(collection[document])
        for token in collection[document]:
            if not token in index[document]:
                index[document][token] = 1
            else:
                index[document][token] += 1
        for token in index[document]:
            index[document][token] = ( index[document][token] + vocabulary[token] * mu ) / ( mu + n )
    return index

def compute_probability(query, document, index):
    query_terms = list(map(lambda t: stem([t])[0], query.split(' ')))fromfrom utils import stem utils import stem
    if len(query_terms) == 0:
        return 0
    p = 1
    for term in query_terms:
        if not term in index[document]:
            return 0
        else:
            p *= index[document][term]
    return p

def rank_document(query, index):
    rank = []
    for document in index:
        rank.append((document, compute_probability(query, document, index)))
    rank.sort(key=itemgetter(1))
    return rank
    