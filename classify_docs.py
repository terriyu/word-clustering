# Classify documents

from __future__ import division
import numpy as np
import argparse, json, time
import process_util

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups

import sklearn.metrics
import sklearn.cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# Example usage: python classify_docs.py --train train.json --test test.json --clusters clusters.json

##### GLOBAL CONSTANTS #####

# Switches for whether or not to run cross-validation (very time-consuming)
CV_LR_FLAG = False
CV_SVM_FLAG = False

##### PARSER #####

parser = argparse.ArgumentParser(description='Classify documents', add_help=False)

required_args = parser.add_argument_group('Required arguments')

optional_args = parser.add_argument_group('Optional arguments')

help_arg = parser.add_argument_group('Help')

required_args.add_argument('--train', required=True, help='JSON file containing cleaned training documents')
required_args.add_argument('--test', required=True, help='JSON file containing cleaned test documents')
required_args.add_argument('--clusters', required=True, help='JSON file containing clusters')

optional_args.add_argument('--mallet', required=False, action='store_true', help='Indicates the cluster file is in MALLET word topic format')
optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

##### FUNCTIONS #####

def cv_error(x, y, clf, method='kfold', num_folds=5, random_iters=10, test_frac=0.2):
    """ Compute cross-validation error and standard deviation of error,
        given input x, output y, and classifier clf

        Default method is k-fold with 5 folds, can also use random re-sampling
    """

    if len(x) != len(y):
        print "Inputs need to be same length as outputs"
        return

    scores = []
    if method == 'kfold':
        # k-fold cross-validation
        cv_iter = sklearn.cross_validation.KFold(len(x), n_folds=num_folds, shuffle=True)
    elif method == 'random':
        # Random resampling
        cv_iter = sklearn.cross_validation.ShuffleSplit(len(x), n_iter=random_iters, test_size=test_frac)
    # Estimate validation error and its standard deviation
    for train, test in cv_iter:
        x_train, x_test, y_train, y_test = x[train,:], x[test,:], y[train], y[test]
        clf.fit(x_train, y_train)
        scores.append(1 - sklearn.metrics.accuracy_score(y_test, clf.predict(x_test)))

    return np.mean(scores), np.std(scores)

def extract_features(docs, vocab_set, vocab_map, word_cluster_map, cluster_id_map):
    """ Extract features from documents given vocabulary and clusters

        Mappings vocab_map, word_cluster_map, and cluster_id_map provide ordering
        for features in the feature vector

        vocab_map maps token to index in word feature vector
        word_cluster_map maps token to cluster ids
        cluster_id_map maps cluster id to index in cluster feature vector
    """

    word_features = np.zeros((len(docs), len(vocab_set)))
    cluster_features = np.zeros((len(docs), len(cluster_id_map)))
    full_features = np.zeros((len(docs), len(vocab_set) + len(cluster_id_map)))

    docs_skipped = 0
    for i, tokens in enumerate(docs):

        word_dist = np.zeros(len(vocab_set))
        cluster_dist = np.zeros(len(cluster_id_map))

        if vocab_set.intersection(tokens):
            for token in tokens:
                if token in vocab_set:
                    # Increment word count
                    idx = vocab_map[token]
                    word_dist[idx] += 1

                    # Increment cluster id counts
                    ids = word_cluster_map[token]
                    for cid in ids:
                        idx = cluster_id_map[cid]
                        cluster_dist[idx] += 1

            # Normalize distributions
            word_dist = word_dist / np.sum(word_dist)
            cluster_dist = cluster_dist / np.sum(cluster_dist)
        else:
            # Increment number of training docs with no words in vocab
            docs_skipped += 1

        word_features[i, :] = word_dist
        cluster_features[i, :] = cluster_dist
        full_features[i, :] = np.hstack((word_dist, cluster_dist))

    return word_features, cluster_features, full_features, docs_skipped

##### MAIN SCRIPT #####

ti = time.time()

if args.verbose: print "Loading documents..."

# Load documents
with open(args.train, 'r') as input_file:
    train_docs = json.load(input_file)

with open(args.test, 'r') as input_file:
    test_docs = json.load(input_file)

# Load clusters
if args.verbose: print "Loading clusters..."

if args.mallet:
    clusters, clusters_with_counts, clusters_words = process_util.create_mallet_clusters(args.clusters, 10, vocab)
else:
    with open(args.clusters, 'r') as input_file:
        cluster_data = json.load(input_file)

    clusters = cluster_data['clusters_by_count']
    word_cluster_map = cluster_data['word_cluster_map']
    cluster_ids = cluster_data['cluster_ids']

    # Extract vocabulary from clusters
    vocabulary = []
    for c in clusters:
        vocabulary.extend(c)
    vocabulary.sort()

    # Turn vocabulary list into set
    vocab_set = set(vocabulary)

    # Create mapping from vocabulary word to index
    # Indices correspond to vector word_dist
    vocab_map = {word: idx for idx, word in enumerate(vocabulary)}

    # Create from cluster id to index
    # Indices correspond to vector cluster_dist
    cluster_id_map = {c_id: idx for idx, c_id in enumerate(cluster_ids)}

# Extract features

# Extract features for training set
if args.verbose: print "Extracting training features..."

train_word_features, train_cluster_features, train_full_features, train_docs_skipped = extract_features(train_docs, vocab_set, vocab_map, word_cluster_map, cluster_id_map)

print "Number of training docs with no words in vocab (skipped feature extraction) = %s" % train_docs_skipped

# Extract features for test set
if args.verbose: print "Extracting test features..."

test_word_features, test_cluster_features, test_full_features, test_docs_skipped = extract_features(test_docs, vocab_set, vocab_map, word_cluster_map, cluster_id_map)

print "Number of test docs with no words in vocab (skipped feature extraction) = %s" % test_docs_skipped

# Train classifiers

# Fetch targets for training and test sets
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
train_targets = newsgroups_train.target
test_targets = newsgroups_test.target

##### CROSS-VALIDATION #####

# Cross-validation for logistic regression
if CV_LR_FLAG:
    c_vals_lr = 10**np.arange(-3.0, 8.0, 1.0)

    err_word_lr = np.empty(len(c_vals_lr))
    std_word_lr = np.empty(len(c_vals_lr))
    err_cluster_lr = np.empty(len(c_vals_lr))
    std_cluster_lr = np.empty(len(c_vals_lr))
    err_full_lr = np.empty(len(c_vals_lr))
    std_full_lr = np.empty(len(c_vals_lr))

    print "Cross-validation for logistic regression"

    for i, c_val in enumerate(c_vals_lr):
        print "C = %s" % c_val
        clf_word = LogisticRegression(C=c_val)
        err_word_lr[i], std_word_lr[i] = cv_error(train_cluster_features, train_targets, clf_word)
        print "Word features - cv err = %s (std = %s)" % (err_word_lr[i], std_word_lr[i])
        clf_cluster = LogisticRegression(C=c_val)
        err_cluster_lr[i], std_cluster_lr[i] = cv_error(train_cluster_features, train_targets, clf_word)
        print "Cluster features - cv err = %s (std = %s)" % (err_cluster_lr[i], std_cluster_lr[i])
        clf_full = LogisticRegression(C=c_val)
        err_full_lr[i], std_full_lr[i] = cv_error(train_full_features, train_targets, clf_full)
        print "Full features - cv err = %s (std = %s)" % (err_full_lr[i], std_full_lr[i])

    plt.figure(1)
    plt.plot(np.log10(c_vals_lr), err_word_lr)
    plt.xlabel('log10(C)')
    plt.ylabel('Cross-validation error')
    plt.title('Logistic regression - Word features')

    plt.figure(2)
    plt.plot(np.log10(c_vals_lr), err_cluster_lr)
    plt.xlabel('log10(C)')
    plt.ylabel('Cross-validation error')
    plt.title('Logistic regression - Cluster features')

    plt.figure(3)
    plt.plot(np.log10(c_vals_lr), err_full_lr)
    plt.xlabel('log10(C)')
    plt.ylabel('Cross-validation error')
    plt.title('Logistic regression - Full features')

# Cross-validation for SVM
if CV_SVM_FLAG:
    c_vals_svm = 10**np.arange(-3.0, 8.0, 1.0)

    err_word_svm = np.empty(len(c_vals_svm))
    std_word_svm = np.empty(len(c_vals_svm))
    err_cluster_svm = np.empty(len(c_vals_svm))
    std_cluster_svm = np.empty(len(c_vals_svm))
    err_full_svm = np.empty(len(c_vals_svm))
    std_full_svm = np.empty(len(c_vals_svm))

    print "Cross-validation for SVM"

    for i, c_val in enumerate(c_vals_svm):
        print "C = %s" % c_val
        clf_word = LinearSVC(C=c_val)
        err_word_svm[i], std_word_svm[i] = cv_error(train_cluster_features, train_targets, clf_word)
        print "Word features - cv err = %s (std = %s)" % (err_word_svm[i], std_word_svm[i])
        clf_cluster = LinearSVC(C=c_val)
        err_cluster_svm[i], std_cluster_svm[i] = cv_error(train_cluster_features, train_targets, clf_word)
        print "Cluster features - cv err = %s (std = %s)" % (err_cluster_svm[i], std_cluster_svm[i])
        clf_full = LinearSVC(C=c_val)
        err_full_svm[i], std_full_svm[i] = cv_error(train_full_features, train_targets, clf_full)
        print "Full features - cv err = %s (std = %s)" % (err_full_svm[i], std_full_svm[i])

    plt.figure(1)
    plt.plot(np.log10(c_vals_svm), err_word_svm)
    plt.xlabel('log10(C)')
    plt.ylabel('Cross-validation error')
    plt.title('SVM - Word features')

    plt.figure(2)
    plt.plot(np.log10(c_vals_svm), err_cluster_svm)
    plt.xlabel('log10(C)')
    plt.ylabel('Cross-validation error')
    plt.title('SVM - Cluster features')

    plt.figure(3)
    plt.plot(np.log10(c_vals_svm), err_full_svm)
    plt.xlabel('log10(C)')
    plt.ylabel('Cross-validation error')
    plt.title('SVM - Full features')

##### NAIVE BAYES #####

print "Multinomial Naive Bayes..."

ti = time.time()

# Train classifiers

clf_word_nb = MultinomialNB()
clf_word_nb.fit(train_word_features, train_targets)
train_word_accuracy = sklearn.metrics.accuracy_score(train_targets, clf_word_nb.predict(train_word_features))

print "Training accuracy with word features = %s" % train_word_accuracy

clf_cluster_nb = MultinomialNB()
clf_cluster_nb.fit(train_cluster_features, train_targets)
train_cluster_accuracy = sklearn.metrics.accuracy_score(train_targets, clf_cluster_nb.predict(train_cluster_features))

print "Training accuracy with cluster features = %s" % train_cluster_accuracy

clf_full_nb = MultinomialNB()
clf_full_nb.fit(train_full_features, train_targets)
train_full_accuracy = sklearn.metrics.accuracy_score(train_targets, clf_full_nb.predict(train_full_features))

print "Training accuracy with full features = %s" % train_full_accuracy

# Evaluate classifiers

test_word_accuracy = sklearn.metrics.accuracy_score(test_targets, clf_word_nb.predict(test_word_features))
print "Test accuracy with word features = %s" % test_word_accuracy

test_cluster_accuracy = sklearn.metrics.accuracy_score(test_targets, clf_cluster_nb.predict(test_cluster_features))
print "Test accuracy with cluster features = %s" % test_cluster_accuracy

test_full_accuracy = sklearn.metrics.accuracy_score(test_targets, clf_full_nb.predict(test_full_features))
print "Test accuracy with full features = %s" % test_full_accuracy

tf = time.time()

if args.verbose: print "Total processing time for Naive Bayes = %s seconds" % (tf-ti)

##### LOGISTIC REGRESSION #####

print "Logistic regression..."

ti = time.time()

# Train classifiers

clf_word_lr = LogisticRegression(C=100)
clf_word_lr.fit(train_word_features, train_targets)
train_word_accuracy = sklearn.metrics.accuracy_score(train_targets, clf_word_lr.predict(train_word_features))

print "Training accuracy with word features = %s" % train_word_accuracy

clf_cluster_lr = LogisticRegression(C=10000)
clf_cluster_lr.fit(train_cluster_features, train_targets)
train_cluster_accuracy = sklearn.metrics.accuracy_score(train_targets, clf_cluster_lr.predict(train_cluster_features))

print "Training accuracy with cluster features = %s" % train_word_accuracy

clf_full_lr = LogisticRegression(C=100)
clf_full_lr.fit(train_full_features, train_targets)
train_full_accuracy = sklearn.metrics.accuracy_score(train_targets, clf_full_lr.predict(train_full_features))

print "Training accuracy with full features = %s" % train_full_accuracy

# Evaluate classifiers

test_word_accuracy = sklearn.metrics.accuracy_score(test_targets, clf_word_lr.predict(test_word_features))
print "Test accuracy with word features = %s" % test_word_accuracy

test_cluster_accuracy = sklearn.metrics.accuracy_score(test_targets, clf_cluster_lr.predict(test_cluster_features))
print "Test accuracy with cluster features = %s" % test_cluster_accuracy

test_full_accuracy = sklearn.metrics.accuracy_score(test_targets, clf_full_lr.predict(test_full_features))
print "Test accuracy with full features = %s" % test_full_accuracy

tf = time.time()

if args.verbose: print "Total processing time for logistic regression = %s seconds" % (tf-ti)

##### SVM #####

print "SVM..."

ti = time.time()

# Train classifiers

clf_word_svm = LinearSVC(C=100)
clf_word_svm.fit(train_word_features, train_targets)
train_word_accuracy = sklearn.metrics.accuracy_score(train_targets, clf_word_svm.predict(train_word_features))

print "Training accuracy with word features = %s" % train_word_accuracy

clf_cluster_svm = LinearSVC(C=100)
clf_cluster_svm.fit(train_cluster_features, train_targets)
train_cluster_accuracy = sklearn.metrics.accuracy_score(train_targets, clf_cluster_svm.predict(train_cluster_features))

print "Training accuracy with cluster features = %s" % train_word_accuracy

clf_full_svm = LinearSVC(C=10)
clf_full_svm.fit(train_full_features, train_targets)
train_full_accuracy = sklearn.metrics.accuracy_score(train_targets, clf_full_svm.predict(train_full_features))

print "Training accuracy with full features = %s" % train_full_accuracy

# Evaluate classifiers

test_word_accuracy = sklearn.metrics.accuracy_score(test_targets, clf_word_svm.predict(test_word_features))
print "Test accuracy with word features = %s" % test_word_accuracy

test_cluster_accuracy = sklearn.metrics.accuracy_score(test_targets, clf_cluster_svm.predict(test_cluster_features))
print "Test accuracy with cluster features = %s" % test_cluster_accuracy

test_full_accuracy = sklearn.metrics.accuracy_score(test_targets, clf_full_svm.predict(test_full_features))
print "Test accuracy with full features = %s" % test_full_accuracy

tf = time.time()

if args.verbose: print "Total processing time for SVM = %s seconds" % (tf-ti)

