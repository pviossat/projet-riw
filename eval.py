from os.path import isfile

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def mAp(QUERIES,results_obtained,query_output, K,model_name,forcebuild= False):
    precisions = []
    for i,query in QUERIES.items():
        print("Computing the mean average precision for query {}".format(query))
        true_results = query_output[str(i)]
        precisions.append(average_precision(results_obtained[str(i)],true_results,K,model_name,str(i),forcebuild))
    return np.mean(precisions)


def average_precision(relevant_doc_ids, true_result ,K,model_name,query_number,forcebuild=False):
    results = get_recall_precision(relevant_doc_ids, true_result,model_name,query_number,forcebuild)
    means = np.zeros(K)
    counter = 0
    for value in np.arange(0, 1, 1 / K):
        means[counter] = interpolate_precision(value, results)
        counter += 1
    return np.mean(means)


def get_recall_precision(relevant_doc_ids, true_result,model_name,query_number,forcebuild=False):
    name = model_name+query_number
    if not forcebuild and isfile(name):
        filehandler = open(model_name, 'rb')
        precision_recall = pickle.load(filehandler)
    else:
        precision_recall = list(map(lambda x: evaluate(x, true_result, relevant_doc_ids), tqdm(range(1, len(relevant_doc_ids)))))
        filehander = open(name, 'wb')
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


def draw_interpolate_recall_precision_curve(relevant_doc_ids, true_result,model_name,K,query_number):
    results = get_recall_precision(relevant_doc_ids, true_result,model_name,query_number,forcebuild=False)
    interpolates = np.zeros(K)
    counter = 0
    for value in np.arange(0, 1, 1/K):
        interpolates[counter] = interpolate_precision(value, results)
        counter += 1
    plt.plot(interpolates, np.arange(0, 1, 1/K))



