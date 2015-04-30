# Computes variation of information between two clusters

from __future__ import division
import numpy as np
import argparse, json

# Example usage: python evaluate_vi.py --pmi_lda pmi_clusters.json word_topic_counts.txt

##### PARSER #####

parser = argparse.ArgumentParser(description='Evaluate variation of information between two clusters', add_help=False)

ex_group = parser.add_argument_group(title='Mutually exclusive arguments (one is required)')
ex_args = ex_group.add_mutually_exclusive_group(required=True)

required_args = parser.add_argument_group('Required arguments')

optional_args = parser.add_argument_group('Optional arguments')

help_arg = parser.add_argument_group('Help')

ex_args.add_argument('--pmi_lda', nargs=2, help='Compute variation of information between PMI and LDA clusters (args = JSON PMI cluster file, MALLET word topic keywords file)')
ex_args.add_argument('--lda_lda', nargs=2, help='Compute variation of information between LDA and LDA clusters (args = MALLET word topic keywords file1, MALLET word topics keyword file2')

required_args.add_argument('--vocab', required=True, help='JSON file containing vocabulary')
required_args.add_argument('--n_clusters', required=True, type=int, help='Number of clusters to use for LDA')

optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

##### EVALUATION METHODS #####

def create_mallet_clusters(filename, num_clusters, vocab):
    """ Create clusters corresponding to MALLET word topic counts file,
        given the number of clusters, also return the list of words in the MALLET clusters

        Only include a word in the MALLET clusters if it is in our PMI vocabulary
    """
    # Words that appear in the MALLET clusters
    cluster_words = []
    # Clusters corresponding to MALLET word topic counts
    clusters = [None] * num_clusters
    # Same as above but with counts
    clusters_counts = [None] * num_clusters

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        tokens = line.strip().split()
        # Extract word and highest count from MALLET file
        # Highest count has form i:j where i is the cluster id
        # and j is the number of counts
        word, highest_count  = tokens[1:3]
        if word in vocab:
            cluster_words.append(word)
            cluster_idx, count = [int(s) for s in highest_count.split(':')]
            if clusters[cluster_idx] is None:
                clusters[cluster_idx] = [word]
                clusters_counts[cluster_idx] = [(word, count)]
            else:
                clusters[cluster_idx].append(word)
                clusters_counts[cluster_idx].append((word, count))

    for c in clusters_counts:
        c.sort(key=lambda x: x[1], reverse=True)

    return clusters, clusters_counts, cluster_words

def calculate_VI(clusters1, clusters2):
    """ Calculate variation of information between two sets of clusters
    """
    if len(clusters1) != len(clusters2):
        print "Warning: number of clusters does not match"

    # Total number of elements in clusters
    n1 = sum([len(c) for c in clusters1])
    n2 = sum([len(c) for c in clusters2])

    if n1 != n2:
        print "Warning: Number of elements in clusters do not match"
        print "Elements in cluster1 = %d, elements in cluster2 = %d" % (n1, n2)

    n = n1
    vi = 0.0
    for i in range(len(clusters1)):
        for j in range(len(clusters2)):
            # Compute probabilities for single clusters
            p_i = len(clusters1[i])/n
            q_j = len(clusters2[j])/n
            # Check if clusters i and j have an intersection
            set_i = set(clusters1[i])
            set_j = set(clusters2[j])
            intersection_ij =  set_i.intersection(set_j)
            if intersection_ij:
                # If intersection exists, r_ij is nonzero and
                # the contribution to VI is nonzero
                r_ij = len(intersection_ij)/n
                vi -= r_ij * (np.log2(r_ij/p_i) + np.log2(r_ij/q_j))

    return vi

##### MAIN SCRIPT #####

# Load vocabulary
if args.verbose:
    print "Loading vocabulary..."

with open(args.vocab, 'r') as input_file:
    data = json.load(input_file)

vocab = set(data['vocab'])

# Loading clusters
if args.verbose:
    print "Loading clusters..."

if args.pmi_lda is not None:
    with open(args.pmi_lda[0], 'r') as input_file:
        pmi_clusters_data = json.load(input_file)
    clusters1 = pmi_clusters_data['clusters']
    clusters2, clusters2_counts, clusters2_words = create_mallet_clusters(args.pmi_lda[1], args.n_clusters, vocab)
else:
    clusters1, clusters1_counts, clusters1_words = create_mallet_clusters(args.lda_lda[0], args.n_clusters, vocab)
    clusters2, clusters2_counts, clusters2_words = create_mallet_clusters(args.lda_lda[1], args.n_clusters, vocab)

if args.verbose:
    print "Cluster sizes for cluster 1"
    print [len(c) for c in clusters1]
    print "Cluster sizes for cluster 2"
    print [len(c) for c in clusters2]

vi = calculate_VI(clusters1, clusters2)

print "Variation of information between clusters = %s" % vi
