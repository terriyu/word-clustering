# Calculate coherence of clusters 

from __future__ import division
import numpy as np
import argparse, json
import process_util

# Example usage: python evaluate_coherence.py --clusters clusters.json --counts counts.json

##### PARSER #####

parser = argparse.ArgumentParser(description='Evaluate coherence of a cluster', add_help=False)

required_args = parser.add_argument_group('Required arguments')

optional_args = parser.add_argument_group('Optional arguments')

help_arg = parser.add_argument_group('Help')

required_args.add_argument('--clusters', required=True, help='JSON or MALLET file containing clusters')
required_args.add_argument('--counts', required=True, help='JSON file containing counts and possibly score table')
required_args.add_argument('--N', required=True, type=int, help='Top N words to use from each cluster')

# This should be a mutually exclusive argument
# Add argument for type of score
optional_args.add_argument('--use_table', required=False, action='store_true', help='Use pre-computed score table, instead of computing scores from counts')

optional_args.add_argument('--mallet', required=False, action='store_true', help='Indicates the cluster file is in MALLET word topic format')
optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

##### EVALUATION METHODS #####

def calculate_mean_coherence(N, clusters, score_table):
    """ Calculate mean pairwise coherence of clusters, given PMI scores 

        Note: Any word pairs not in the score table have their score truncated to zero 
    """
    num_pairs = N*(N-1)//2
    coherence = [] 
    coherence_table = {}
    for c in clusters:
       top_words = sorted(c[:N]) 
       total = 0.0
       for i in range(N):
           for j in range(i+1, len(top_words)):
               num_pairs += 1
               wi = top_words[i]
               wj = top_words[j]  
               if wi in score_table:
                   if wj in score_table[wi]:
                       total += score_table[wi][wj] 
                       pair = tuple(sorted([wi,wj]))
                       coherence_table[pair] = score_table[wi][wj]
       coherence.append(total/num_pairs)

    coherence = np.array(coherence)
    return coherence.mean(), coherence, coherence_table

##### MAIN SCRIPT #####

# Load count data
if args.verbose:
    print "Loading count data..."

with open(args.counts, 'r') as input_file:
    count_data = json.load(input_file)

vocab = count_data['vocab']
doc_single_counts = count_data['single_counts']
score_lookup = count_data['score_table']

# Need to convert flattened keys back to tuple keys
doc_pair_counts = {}
for flat_key, value in count_data['pair_counts'].iteritems():
    tuple_key = tuple(sorted(flat_key.split(',')))
    doc_pair_counts[tuple_key] = value

# Load clusters
if args.verbose:
    print "Loading clusters..."

if args.mallet:
    unsorted_clusters, clusters_counts, clusters_words = process_util.create_mallet_clusters(args.clusters, 10, vocab) 
    sorted_clusters = []
    for c in clusters_counts:
        sorted_clusters.append([x[0] for x in c])
else:
    with open(args.clusters, 'r') as input_file:
        cluster_data = json.load(input_file)

    sorted_clusters = cluster_data['clusters_by_count']

# Calculate coherence of each cluster
if args.verbose:
    print "Calculating coherence..."

coherence_mean, coherence_dist, coherence_table = calculate_mean_coherence(args.N, sorted_clusters, score_lookup)

print "Coherence distribution"
print coherence_dist
print "Mean coherence = %s" % coherence_mean
