import numpy as np
from utils import stem
from collections import Counter
from tqdm import tqdm


def processing_vectorial_query(query, index, stats_collection, weighting_scheme_document, weighting_scheme_query):
    query_terms = list(map(lambda t: stem([t])[0], query.split(' ')))
    nb_doc = stats_collection["nb_doc"]
    vocabulary = index.keys()
    docs_ids = stats_collection.keys()
    relevant_docs = {}
    n_query = 0
    s = dict(map(lambda x: (x, 0), docs_ids))
    n_document = dict(map(lambda x: (x, 0), docs_ids))
    for term in query_terms:
        if term in vocabulary:
            w_query = query_weighting(weighting_scheme_query, query, term, nb_doc, index)
            n_query += w_query ** 2
            postings = index[term].keys()
            for doc_ID in postings:
                w_document = document_weighting(weighting_scheme_document, doc_ID, term, index, nb_doc,
                                                stats_collection)
                s[doc_ID] += w_query * w_document
                n_document[doc_ID] += w_document ** 2
    for doc_ID in docs_ids:
        if s[doc_ID] != 0:
            s[doc_ID] = s[doc_ID] / (n_document[doc_ID] * n_query) ** (1 / 2)
            relevant_docs[doc_ID] = s[doc_ID]
    ordered_relevant_docs = {k: v for k, v in sorted(relevant_docs.items(), key=lambda item: item[1], reverse=True)}
    return ordered_relevant_docs


def query_weighting(weighting_scheme_query, query, term, nb_doc, index):
    if weighting_scheme_query == "BIN":
        return 1
    elif weighting_scheme_query == "FRQ":
        return query.count(term)  # * np.log(nb_doc / len(index[term].keys()))
    else:
        print('Error : Wrong weighting scheme for query')


def document_weighting(weighting_scheme_document, doc_ID, term, index, nb_doc, stats_collection):
    if weighting_scheme_document == "BIN":
        return 1
    elif weighting_scheme_document == "FRQ":
        return get_tf(term, doc_ID, index)
    elif weighting_scheme_document == "TIN":
        return get_tf_normalise(term, doc_ID, index, stats_collection) * get_idf(term, index, nb_doc)
    elif weighting_scheme_document == "TIL":
        return get_tf_logarithmique(term, doc_ID, index) * get_idf(term, index, nb_doc)
    elif weighting_scheme_document == "TILN":
        return get_tf_logarithme_normalise(term, doc_ID, index, stats_collection) * get_idf(term, index, nb_doc)
    else:
        print('Error : Wrong weighting scheme for document')


def get_tf(term, doc_ID, index):
    return index[term][doc_ID][0]


def get_idf(term, index, nb_doc):
    occurences = len(index[term].keys())
    return np.log(nb_doc / occurences)


def get_idf_normalized(term, index, nb_doc):
    occurences = len(index[term].keys())
    return max(0, np.log((nb_doc - occurences) / occurences))


def get_tf_logarithmique(term, doc_ID, index):
    return np.log(index[term][doc_ID][0])


def get_tf_normalise(term, doc_ID, index_frequence, stats_collection):
    return 0.5 + 0.5 * get_tf(term, doc_ID, index_frequence) / stats_collection[doc_ID]["freq_max"]


def get_tf_logarithme_normalise(term, doc_ID, index_frequence, stats_collection):
    tf = get_tf(term, doc_ID, index_frequence)
    if tf > 0:
        return (1 + np.log(tf)) / (1 + np.log(stats_collection[doc_ID]["average_frequencies"]))
    else:
        return 1 / (1 + np.log(stats_collection[doc_ID]["average_frequencies"]))


def get_stats_document(document):
    stats = {}
    counter = Counter(document)
    frequencies = counter.values()
    stats["freq_max"] = max(frequencies)
    stats["unique_terms"] = len(frequencies)
    stats["average_frequencies"] = np.array([counter[k] for k in counter]).mean()
    return stats


def get_stats_collection(collection):
    print("Computing statistics on the entire collection")
    values = tqdm(collection.items())
    stats = dict(map(lambda x: (x[0], get_stats_document(x[0])), values))
    stats["nb_doc"] = sum(map(len, collection.values()))
    return stats
