import numpy as np
from vectorial_model import processing_vectorial_query
import pickle
import matplotlib.pyplot as plt


def mean_average_precision(QUERIES, inverted_index, model, query_output, K):
    average_precision = []
    for query in QUERIES.values():
        average_precision.append(mean_precision(query, inverted_index, model, query_output, K))
    return np.mean(average_precision)


def mean_precision(query, index, model, query_output, K, weighting_schemes, stats_collection):
    results = draw_recall_precision_curve(query, index, model, query_output, weighting_schemes, stats_collection)
    means = np.zeros(K)
    counter = 0
    for value in np.arange(0, 1, 1 / K):
        means[counter] = interpolate_precision(value, results)
        counter += 1
    return np.mean(means)


def draw_recall_precision_curve(query, index, model, query_output, weighting_schemes, stats_collection):
    weighting_scheme_query = weighting_schemes["frequency"]
    weighting_scheme_document = weighting_schemes[model]
    processed = processing_vectorial_query(query, index, stats_collection, weighting_scheme_document,
                                           weighting_scheme_query)
    relevant_docs = list(processed.keys())
    precision_recall = list(map(lambda x: evaluate(x, query_output, relevant_docs), tqdm(range(1, len(relevant_docs)))))
    filehander = open(model, 'wb')
    pickle.dump(precision_recall, filehander)
    return np.array(precision_recall)


def evaluate(doc_id, query_output, ordered_relevant_docs):
    true = set(query_output[:doc_id])
    obtained = set(ordered_relevant_docs[:doc_id])
    false_positives = obtained - true
    true_positives = obtained.intersection(true)
    return precision(true_positives, false_positives), recall(true_positives, query_output)


def precision(true_positives, false_positives):
    tp = len(true_positives)
    fp = len(false_positives)
    return tp / (tp + fp)


def recall(true_positives, query_output):
    return len(true_positives) / len(query_output)


def interpolate_precision(value, results):
    p = results[:, 0]
    r = results[:, 1]
    try:
        interpolation = max(p[np.where(r >= value)])
    except ValueError:
        interpolation = 0
    return interpolation


def draw_interpolate_recall_precision_curve(query, index, model, query_output, weighting_schemes, stats_collection):
    results = draw_recall_precision_curve(query, index, model, query_output, weighting_schemes, stats_collection)
    interpolates = np.zeros(10)
    counter = 0
    for value in np.arange(0, 1, 0.1):
        interpolates[counter] = interpolate_precision(value, results)
        counter += 1
    plt.plot(interpolates, np.arange(0, 1, 0.1))
