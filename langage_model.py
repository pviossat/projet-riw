from utils import stem
from operator import itemgetter

def create_vocabulary(collection):
    vocabulary = dict()
    n = 0
    for tokens in collection.values():
        for token in tokens:
            if not token in vocabulary:
                vocabulary[token] = 0
            vocabulary[token] += 1
            n += 1
    for token in vocabulary:
        vocabulary[token] = vocabulary[token] / n
    return vocabulary

def create_language_model(collection, vocabulary, mu):
    index = dict()
    for document in collection:
        index[document] = dict()
        n = len(collection[document])
        for token in collection[document]:
            if not token in index[document]:
                index[document][token] = 0
            index[document][token] += 1
        for token in index[document]:
            index[document][token] = ( index[document][token] + vocabulary[token] * mu ) / ( mu + n )
        index[document]["$length"] = n
    return index

def compute_probability(query, document, index, vocabulary, mu):
    query_terms = list(map(lambda t: stem([t])[0], query.split(' ')))
    if len(query_terms) == 0:
        return 0
    p = 1
    for term in query_terms:
        if not term in index[document]:
            if not term in vocabulary:
                return 0
            else:
                return vocabulary[term] * mu / ( mu + index[document]["$length"])
        else:
            p *= index[document][term]
    return p

def rank_document(query, index, vocabulary, mu):
    rank = []
    for document in index:
        rank.append((document, compute_probability(query, document, index, vocabulary, mu)))
    rank.sort(key=itemgetter(1), reverse=True)
    return rank
    