# Classify documents

from __future__ import division
import numpy as np
import argparse, json
import process_util

# Example usage: python classify_docs.py 

##### PARSER #####

parser = argparse.ArgumentParser(description='Classify documents', add_help=False)

required_args = parser.add_argument_group('Required arguments')

optional_args = parser.add_argument_group('Optional arguments')

help_arg = parser.add_argument_group('Help')

required_args.add_argument('--docs', required=True, help='JSON file containing cleaned documents')
required_args.add_argument('--clusters', required=True, help='JSON file containing clusters')

optional_args.add_argument('--mallet', required=False, action='store_true', help='Indicates the cluster file is in MALLET word topic format')
optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

##### MAIN SCRIPT #####

# Load docouments
with open(args.docs, 'r') as input_file:
    docs = json.load(input_file)

# Load clusters
if args.verbose:
    print "Loading clusters..."

if args.mallet:
    clusters, clusters_with_counts, clusters_words = process_util.create_mallet_clusters(args.clusters, 10, vocab)
else:
    with open(args.clusters, 'r') as input_file:
        cluster_data = json.load(input_file)

    clusters = cluster_data['clusters_by_count']
    clusters_words = []
    for c in clusters:
        clusters_words.extend(c)

# Extract features
features = [] 
for tokens in docs:
    doc_words = set(tokens)
    doc_len = len(tokens)

    cluster_dist = [] 
    for c in clusters:
        cluster_count = 0
        for token in tokens:
            if token in c:
                cluster_count += 1    
        cluster_dist.append(cluster_count)

    features.append(cluster_dist)
