from os.path import isfile

import numpy as np
from vectorial_model import processing_vectorial_query
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def mean_average_precision(QUERIES, index, model, true_results, K, weighting_schemes, stats_collection,forcebuild=False):
    average_precision = []
    for i,query in QUERIES.items():
        average_precision.append(mean_precision(query, index, model, true_results[str(i)], K, weighting_schemes, stats_collection,forcebuild))
    return np.mean(average_precision)


def mean_precision(query, index, model, true_result, K, weighting_schemes, stats_collection,forcebuild=False):
    results = draw_recall_precision_curve(query, index, model, true_result, weighting_schemes, stats_collection,forcebuild)
    means = np.zeros(K)
    counter = 0
    for value in np.arange(0, 1, 1 / K):
        means[counter] = interpolate_precision(value, results)
        counter += 1
    return np.mean(means)


def draw_recall_precision_curve(query, index, model, true_result, weighting_schemes, stats_collection, forcebuild=False):
    weighting_scheme_query = weighting_schemes["frequency"]
    weighting_scheme_document = weighting_schemes[model]
    if not forcebuild and isfile(model):
        filehandler = open(model, 'rb')
        precision_recall = pickle.load(filehandler)
    else:
        processed = processing_vectorial_query(query, index, stats_collection, weighting_scheme_document,
                                           weighting_scheme_query)
        relevant_docs = list(processed.keys())
        precision_recall = list(map(lambda x: evaluate(x, true_result, relevant_docs), tqdm(range(1, len(relevant_docs)))))
        filehander = open(model, 'wb')
        pickle.dump(precision_recall, filehander)
    return np.array(precision_recall)


def evaluate(doc_id, true_result, ordered_relevant_docs):
    true = set(true_result[:doc_id])
    obtained = set(ordered_relevant_docs[:doc_id])
    false_positives = obtained - true
    true_positives = obtained.intersection(true)
    return precision(true_positives, false_positives), recall(true_positives, true_result)


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
