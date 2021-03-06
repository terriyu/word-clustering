from __future__ import division
from collections import defaultdict
from operator import itemgetter
import numpy as np
import argparse, json, time
import matplotlib.pyplot as plt

# Set up terminal to handle unicode printing
# But only if we are in the terminal, not if we are in IPython
import sys, codecs
try:
    __IPYTHON__
except NameError:
    # If you run this code in IPython, the prompt becomes garbled!!
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    sys.stderr = codecs.getwriter('utf8')(sys.stderr)

# Example usage:
# python cluster_pmi.py --metric mean --method lsh --input sotu_small_test_pmi.json --merges_per_batch 10 --output temp.json --tree_html temp_tree.html --verbose

# Set random seed for LSH random projections
np.random.seed(42)
#np.random.seed(137)
#np.random.seed(28)

##### GLOBAL CONSTANTS #####

# Frequency cutoff for inclusion in top words
# Single count frequency must be larger than this number
# to be included in the list of top words
TOP_WORDS_CUTOFF = 10

# Scoring metrics available
# Note: temporarily take disjunction out of valid metrics,
#       will work on disjunction more later
VALID_METRICS = set(['min', 'max', 'mean', 'geometric', 'harmonic'])

# Clustering methods available
# 'hac' = basic hierachical agglomerative clustering + HAC
# 'most_freq_words' = uses Percy Liang heuristic of clustering with most frequent words + HAC
# 'lsh' = Locality Sensitive Hashing (LSH) + HAC
VALID_METHODS = set(['hac', 'most_freq_words', 'lsh'])

##### PARSER #####

parser = argparse.ArgumentParser(description='Cluster documents using PMI-based metrics.', add_help=False)

required_args = parser.add_argument_group('Required arguments')
optional_args = parser.add_argument_group('Optional arguments')
help_arg = parser.add_argument_group('Help')

required_args.add_argument('--metric', required=True, choices=VALID_METRICS, help='Metric to use for clustering linkage')
required_args.add_argument('--method', required=True, choices=VALID_METHODS, help='Method to use for clustering: "hac", "most_freq_words", "lsh"')
required_args.add_argument('--input', required=True, help='Input JSON file containing pre-computed data, including co-occurence counts and score table')
required_args.add_argument('--output', required=True, help='Output JSON file containing clusters')

optional_args.add_argument('--n_clusters', required=False, default=10, type=int, help='Target number of clusters (default=10)')
optional_args.add_argument('--merges_per_batch', required=False, default=10, type=int, help='Number of greedy merges to perform per iteration (default=10)')
optional_args.add_argument('--n_top_words', required=False, default=20, help='Number of top words in each cluster to display (default=20)')
optional_args.add_argument('--tree_html', required=False, help='Filename for HTML output representing merge tree')
optional_args.add_argument('--disjunction', required=False, default=False, action='store_true', help='Use disjunction for LSH clustering')
optional_args.add_argument('--random', required=False, default=False, action='store_true', help='Use random data for testing LSH clustering')
optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

##### LINKAGE METHODS ######

def max_single_pmi_score(pdict, wlist1, wlist2):
    """ Calculate maximum PMI among all word pairs
        in two lists of words, given pre-computed PMI dictionary

        - If there is no PMI defined for any of the word pairs,
          return -inf
    """
    max_pmi = float("-inf")
    for word1 in wlist1:
        for word2 in wlist2:
            # Enforce alphabetical order in pair
            pair = tuple(sorted([word1, word2]))
            wi = pair[0]
            wj = pair[1]
            if wi in pdict and wj in pdict[wi]:
                if pdict[wi][wj] > max_pmi:
                    max_pmi = pdict[wi][wj]
    return max_pmi

def min_single_pmi_score(pdict, wlist1, wlist2):
    """ Calculate minimum PMI score among all word pairs
        in two word lists, given pre-computed PMI dictionary

        - If there is no PMI defined for any of the word pairs,
          return +inf
    """
    min_pmi = float("inf")
    for word1 in wlist1:
        for word2 in wlist2:
            # Enforce alphabetical order in pair
            pair = tuple(sorted([word1, word2]))
            wi = pair[0]
            wj = pair[1]
            if wi in pdict and wj in pdict[wi]:
                if pdict[wi][wj] < min_pmi:
                   min_pmi = pdict[wi][wj]
    return min_pmi

def mean_ppmi_score(pdict, wlist1, wlist2):
    """ Calculate mean positive PMI over all word pairs in two word lists,
        given pre-computed PMI dictionary

        - Any negative PMI is truncated to zero
        - If there is no PMI defined for any of the word pairs,
          return -inf
    """
    total_ppmi = None
    for word1 in wlist1:
        for word2 in wlist2:
            # Enforce alphabetical order in pair
            pair = tuple(sorted([word1, word2]))
            wi, wj = pair
            if wi in pdict and wj in pdict[wi]:
                if total_ppmi is None:
                    total_ppmi = 0
                # Any negative PMIs are considered to be 0
                if pdict[wi][wj] > 0:
                    total_ppmi += pdict[wi][wj]
    if total_ppmi is not None:
        return total_ppmi / (len(wlist1)*len(wlist2))
    else:
        return float("-inf")

def geometric_pmi_score(pdict, wlist1, wlist2):
    """ Calculate geometric mean of PMI over all word pairs
        in two word lists, given pre-computed PMI dictionary

        - If geometric PMI is undefined, return -inf
        - The geometric mean is undefined if:
            - Any of the PMIs are negative
            - None of the word pairs have a defined PMI
    """
    product_pmi = None
    for word1 in wlist1:
        for word2 in wlist2:
            # Enforce alphabetical order in pair
            pair = tuple(sorted([word1, word2]))
            wi, wj = pair
            if wi in pdict and wj in pdict[wi]:
                if product_pmi is None:
                    product_pmi = 1
                pmi = pdict[wi][wj]
                # Check if PMI is negative
                if pmi > 0:
                    product_pmi *= pmi
                else:
                    product_pmi = float("-inf")
                    break
        # If PMI is negative, break out of the loop completely
        if product_pmi == float("-inf"):
            break
    if product_pmi is None:
        # None of the word pairs had a defined PMI
        return float("-inf")
    elif product_pmi == float("-inf"):
        # At least one word pair had a negative PMI
        return float("-inf")
    else:
        return product_pmi ** (1/len(wlist1)/len(wlist2))

def harmonic_epmi_score(pdict, wlist1, wlist2):
    """ Calculate harmonic mean of exponentiated PMI over all word pairs
        in two word lists, given pre-computed PMI dictionary

        - If harmonic ePMI is undefined, return -inf
    """
    total_recip_epmi = None
    # Number of pairs for which PMI exists
    N = 0
    for word1 in wlist1:
       for word2 in wlist2:
            # Enforce alphabetical order in pair
            pair = tuple(sorted([word1, word2]))
            wi, wj = pair
            if wi in pdict and wj in pdict[wi]:
                if total_recip_epmi is None:
                    total_recip_epmi = 0
                total_recip_epmi += 1/(2**pdict[wi][wj])
                N += 1
    if total_recip_epmi is not None:
        return N/total_recip_epmi
    else:
        return float("-inf")

def disjunction_pmi_score(docs, wset1, wset2):
    """ Calculate PMI of two sets of words based on disjunction
        for a given set of documents

        - Uses logarithm base 2
        - If counts of wset1 and wset2 are less than cutoff or
          if all the counts are zero, return -inf
    """
    count_or1 = 0
    count_or2 = 0
    count_and = 0
    for doc in docs:
        doc_words = set(doc['ctext'].split())
        # Do any of the words in wset1 occur in the document?
        w1_or_doc = (len(wset1.intersection(doc_words)) != 0)
        # Do any of the words in wset2 occur in the document?
        w2_or_doc = (len(wset2.intersection(doc_words)) != 0)
        if w1_or_doc:
            count_or1 += 1
        if w2_or_doc:
            count_or2 += 1
        # Does at least 1 word in wset1 and 1 word in wset2 occur in the documents?
        if w1_or_doc and w2_or_doc:
            count_and += 1
    num_docs = len(docs)
    if (count_and != 0) and (count_or1 != 0) and (count_or2 != 0):
        # Check if counts for wset1 and wset2 are above cutoff
        if (count_or1 > VOCAB_CUTOFF) and (count_or2 > VOCAB_CUTOFF):
            disj_pmi = np.log2(num_docs*count_and/(count_or1*count_or2))
        else:
            disj_pmi = float("-inf")
    else:
        # No documents contain any of these words!
        # Set score to -inf
        disj_pmi = float("-inf")
    return disj_pmi

###### CLUSTERING METHODS ######

def score_clusters(pdict, metric, c1, c2):
    """ Calculate score with respect to clusters c1 and c2 using specified metric
    """
    if metric == 'max':
        score = max_single_pmi_score(pdict, c1, c2)
    elif metric == 'min':
        score = min_single_pmi_score(pdict, c1, c2)
    elif metric == 'mean':
        score = mean_ppmi_score(pdict, c1, c2)
    elif metric == 'geometric':
        score = geometric_pmi_score(pdict, c1, c2)
    elif metric == 'harmonic':
        score = harmonic_epmi_score(pdict, c1, c2)
    else:
        print 'No known coherence metric specified'
        return

    return score

def generate_score_table_hac(pdict, clusters, metric):
    """ Generate score table for all pairwise combinations in clusters,
        according to metric, for hierachical agglomerative clustering method

        Score table has form ((i,j), score) where i and j are cluster indices
    """
    ids = clusters.keys()
    cluster_size = len(ids)

    # List of cluster id pairs and corresponding scores
    candidates = []

    # Calculate initial score table
    for i in range(cluster_size):
        for j in range(i+1,cluster_size):
            score = score_clusters(pdict, metric, clusters[ids[i]], clusters[ids[j]])
            # Sort cluster IDs in ascending order
            pair = tuple(sorted([ids[i], ids[j]]))
            candidates.append((pair, score))

    return candidates

def generate_score_table_lsh(pdict, clusters, buckets, metric):
    """ Generate score table for all pairwise combinations in clusters,
        according to metric, for locality sensitive hashing (LSH) method

        Score table has form ((i,j), score) where i and j are cluster indices
    """
    # List of cluster id pairs and corresponding scores
    candidates = []
    # Keep track of which cluster id pairs have already had scores computed
    pairs_computed = set()

    for cids_set in buckets.itervalues():
        # Convert set to list, so we can order the entries and iterate over them
        cids = list(cids_set)
        num_clusters_in_bucket = len(cids)
        if num_clusters_in_bucket > 1:
            for i in range(num_clusters_in_bucket):
                for j in range(i+1,num_clusters_in_bucket):
                    if not pairs_computed.intersection([cids[i], cids[j]]):
                        score = score_clusters(pdict, metric, clusters[cids[i]], clusters[cids[j]])
                        # Sort cluster IDs in ascending order
                        pair = tuple(sorted([cids[i], cids[j]]))
                        candidates.append((pair, score))
                        pairs_computed.add(pair)

    return candidates

def sort_score_table(score_table, metric, stable):
    """ Sort score table

        stable is a Boolean flag indicating whether the sort should use
        stable sort

        Stable sort: For most metrics, if two candidates have the same score,
                     the one with the higher cluster id pair comes first
                     If the metric is 'min', it's the opposite
    """
    if metric != 'min':
        # Sort so that highest scores are at beginning of list
        if stable:
            # Sort by score, then by cluster id pair
            score_table.sort(key=itemgetter(1, 0), reverse=True)
        else:
            # Only sort by score
            score_table.sort(key=itemgetter(1), reverse=True)
    else:
        # Sort so that lowest scores are at beginning of list
        if stable:
            # Sort by score, then by cluster id pair
            score_table.sort(key=itemgetter(1, 0), reverse=False)
        else:
            # Only sort by score
            score_table.sort(key=itemgetter(1), reverse=False)

    return score_table

def compute_buckets(num_bits, proj_mat, clusters, doc_id_vecs):
    """ Hash clusters into LSH buckets

        The feature vector hashed is the doc id vector for a word in the
        cluster.

        This method takes a conservative approach where for a given cluster,
        its cluster ID is placed in any bucket corresponding to the hash of
        any word in the cluster.  Thus, a cluster ID can be in multiple
        buckets.

        Note this method computes a new set of buckets from scratch and does
        not use caching or compression.
    """
    # LSH buckets
    buckets = defaultdict(set)
    # Convenient lookup table -- tells us set of buckets where each
    # cluster id is located
    cid_to_hash = defaultdict(set)
    for cid, c in clusters.iteritems():
        # Iterate over all words in the cluster
        for w in c:
            # Compute random projections
            proj_hash = np.zeros(num_bits, dtype=np.uint8)
            proj_hash[np.dot(proj_mat[:num_bits,:], doc_id_vecs[w]) > 0] = 1
            # Convert hash from array to bitstring
            proj_hash_str = ''.join([str(x) for x in proj_hash.tolist()])
            # Update buckets and cluster id-to-hash mapping
            buckets[proj_hash_str].add(cid)
            cid_to_hash[cid].add(proj_hash_str)

    return cid_to_hash, buckets

def compute_buckets_disjunction(num_bits, proj_mat, clusters, doc_id_vecs):
    """ Hash clusters into LSH buckets

        Here the feature vector hashed is the disjunction of the doc id vectors
        corresponding to the words in the cluster. Specifically, the
        disjunction is calculated by ORing together all the doc id vectors.

        Note this method computes a new set of buckets from scratch and does
        not use caching or compression.
    """
    # LSH buckets
    buckets = defaultdict(set)
    # Convenient lookup table -- tells us set of buckets where each
    # cluster id is located
    cid_to_hash = defaultdict(set)
    for cid, c in clusters.iteritems():
        # OR the doc id vectors for each word in cluster
        or_doc_id_vec = np.zeros(num_docs, dtype=np.uint8)
        for w in c:
            or_doc_id_vec = np.logical_or(or_doc_id_vec, doc_id_vecs[w])
        # Calculate hash from random projections of OR'd doc id vector
        proj_hash = np.zeros(num_bits, dtype=np.uint8)
        proj_hash[np.dot(proj_mat[:num_bits,:], or_doc_id_vec) > 0] = 1
        # Convert hash to bitstring
        proj_hash_str = ''.join([str(x) for x in proj_hash.tolist()])
        # Update buckets and cluster id-to-hash mapping
        buckets[proj_hash_str].add(cid)
        cid_to_hash[cid].add(proj_hash_str)

    return cid_to_hash, buckets

def cache_buckets(max_bits, proj_mat, clusters, doc_id_vecs, use_disjunction):
    """ Compute LSH buckets for number of bits from 1 to max_bits and
        cache in dictionary
    """
    # Calculate buckets corresponding to maximum number of hash bits
    if use_disjunction:
        max_cid_to_hash, max_buckets = compute_buckets_disjunction(max_bits, proj_mat, clusters, doc_id_vecs)
    else:
        max_cid_to_hash, max_buckets = compute_buckets(max_bits, proj_mat, clusters, doc_id_vecs)

    buckets_cached = {}
    cid_to_hash_cached = {}
    buckets_cached[max_bits] = max_buckets
    cid_to_hash_cached[max_bits] = max_cid_to_hash

    for i in range(max_bits-1, 0, -1):
        reduced_buckets = defaultdict(set)
        reduced_cid_to_hash = defaultdict(set)
        # Compute permutation for random selection of indices
        perm = np.arange(max_bits)
        np.random.shuffle(perm)
        # Random indices to extract
        random_idx = perm[:i]
        for hash_str, cids_set in max_buckets.iteritems():
            # Convert hash to NumPy array
            hash_array = np.array(list(hash_str), dtype=np.uint8)
            # Randomly extract num_bits from hash and convert to bitstring
            # This is the compressed hash bitstring
            hash_array = hash_array[random_idx]
            reduced_hash_str = ''.join([str(x) for x in hash_array.tolist()])
            # Update buckets and cluster id-to-hash mapping
            reduced_buckets[reduced_hash_str].update(cids_set)
            for cid in cids_set:
                reduced_cid_to_hash[cid].add(reduced_hash_str)
        buckets_cached[i] = reduced_buckets
        cid_to_hash_cached[i] = reduced_cid_to_hash

    return cid_to_hash_cached, buckets_cached

def compress_buckets(num_bits, max_buckets):
    """ Compress clusters into fewer buckets, reduce hash to fewer bits

        Uses the cached data for buckets calculated with (user-defined)
        maximum number of bits (max_buckets)

        Randomly selects num_bits from the hash for each bucket in order to
        compress the buckets

        Note: this function seems to be very slow, so it's not used currently
    """
    # Compressed LSH buckets
    comp_buckets = defaultdict(set)
    # Lookup table for cluster ids, corresponding to compressed buckets
    comp_cid_to_hash = defaultdict(set)

    for hash_str, cids_set in max_buckets.iteritems():
        # Convert hash to NumPy array
        hash_array = np.array(list(hash_str), dtype=np.uint8)
        # Permute hash
        # Note: random.shuffle() seems faster than random.choice()
        np.random.shuffle(hash_array)
        # Randomly extract num_bits from hash and convert to bitstring
        # This is the compressed hash bitstring
        hash_array = hash_array[:num_bits]
        comp_hash_str = ''.join([str(x) for x in hash_array.tolist()])
        # Update buckets and cluster id-to-hash mapping
        comp_buckets[comp_hash_str].update(cids_set)
        for cid in cids_set:
            comp_cid_to_hash[cid].add(comp_hash_str)

    return comp_cid_to_hash, comp_buckets

def bucket_stats(buckets):
    """ Compute statistics corresponding to LSH bucket distribution
    """
    bucket_dist = np.array([len(x) for x in buckets.values() if len(x) > 1], dtype=np.uint64)
    comparisons_dist = np.array([x*(x-1)//2 for x in bucket_dist], dtype=np.uint64)

    # Mean number of cluster per bucket
    mean_bucket_size = np.mean(bucket_dist)
    # Median number of cluster in buckets
    median_bucket_size = np.median(bucket_dist)
    # Number of pairwise cluster comparisons corresponding to buckets
    num_pairwise_comparisons = np.sum(comparisons_dist)

    # Return stats in alphabetical order by variable name
    return mean_bucket_size, median_bucket_size, num_pairwise_comparisons

def update_buckets(cm1, cm2, id_next, buckets, cid_to_hash, doc_id_vecs, num_bits, proj_mat, clusters, use_disjunction):
    # Remove ids for clusters that have been merged
    # If we are using the conservative method, the new cluster
    # inherits the hashes from the two clusters that were merged to
    # form it
    for h in cid_to_hash[cm1]:
        buckets[h].remove(cm1)
        if not use_disjunction:
            buckets[h].add(id_next)

    for h in cid_to_hash[cm2]:
        buckets[h].remove(cm2)
        if not use_disjunction:
            buckets[h].add(id_next)

    # For disjunction method, hash the new cluster and update the
    # buckets / cluster ID lookup table accordingly
    if use_disjunction:
        or_doc_id_vec = np.zeros(num_docs, dtype=np.uint8)
        for w in clusters[id_next]:
            or_doc_id_vec = np.logical_or(or_doc_id_vec, doc_id_vecs[w])
        # Calculate hash from random projections of OR'd doc id vector
        proj_hash = np.zeros(num_bits, dtype=np.uint8)
        proj_hash[np.dot(proj_mat[:num_bits,:], or_doc_id_vec) > 0] = 1
        # Convert hash to bitstring
        proj_hash_str = ''.join([str(x) for x in proj_hash.tolist()])
        # Update buckets and cluster id-to-hash mapping
        buckets[proj_hash_str].add(id_next)
        cid_to_hash[id_next].add(proj_hash_str)
    else:
        # For the conservative method, the new cluster simply inherits
        # all the hashes from two clusters that were merged to form it,
        # so update the cluster id lookup table accordingly
        cid_to_hash[id_next] = cid_to_hash[cm1].union(cid_to_hash[cm2])

    # Delete cluster ids corresponding to merged clusters from
    # cluster id lookup table
    del cid_to_hash[cm1]
    del cid_to_hash[cm2]

    return cid_to_hash, buckets

def next_buckets(num_bits, proj_mat, clusters, doc_id_vecs, buckets, cid_to_hash, criterion_function, use_disjunction, cache_lsh, buckets_cached=None, cid_to_hash_cached=None):
    """ Return next set of buckets that satisfy the given criterion
    """
    # Calculate bucket statistics
    mean_bucket_size, median_bucket_size, num_pairwise_comparisons = bucket_stats(buckets)

    print "Number of bits in LSH hash = %d" % num_bits
    print "Mean bucket size = %s, median bucket size = %s, number of pairwise comparisons = %d" % (mean_bucket_size, median_bucket_size, num_pairwise_comparisons)

    bucket_size_criterion = criterion_function(median_bucket_size, mean_bucket_size, num_pairwise_comparisons)

    while not bucket_size_criterion and num_bits >= 1:
        if num_bits == 1:
            # Set number of bits to 0
            num_bits = 0

            # Print statistics
            b = len(buckets['0'].union(buckets['1']))
            print "Number of bits in LSH hash = %d" % num_bits
            print "Median bucket size = %s, mean bucket size = %s, number of pairwise comparisons = %d" % (b, b, b*(b-1)/2)

            # Set buckets and lookup table to empty
            # Since there is only one bucket, stop using hash tables and
            # use conventional hierarchical clustering
            buckets = {}
            cid_to_hash = {}

            break

        # Reduce number of bits in hash
        num_bits -= 1

        # Calculate buckets
        if cache_lsh:
            print "cache lsh"
            cid_to_hash = cid_to_hash_cached[num_bits]
            buckets = buckets_cached[num_bits]
        else:
            proj_mat = np.random.rand(num_bits, num_docs) - 0.5
            if use_disjunction:
                cid_to_hash, buckets = compute_buckets_disjunction(num_bits, proj_mat, clusters, doc_id_vecs)
            else:
                cid_to_hash, buckets = compute_buckets(num_bits, proj_mat, clusters, doc_id_vecs)

        # Calculate bucket statistics
        mean_bucket_size, median_bucket_size, num_pairwise_comparisons = bucket_stats(buckets)

        print "Number of bits in LSH hash = %d" % num_bits
        print "Mean bucket size = %s, median bucket size = %s, number of pairwise comparisons = %d" % (mean_bucket_size, median_bucket_size, num_pairwise_comparisons)
        bucket_size_criterion = criterion_function(median_bucket_size, mean_bucket_size, num_pairwise_comparisons)

    return num_bits, cid_to_hash, buckets

def lsh_merge(pdict, doc_id_vecs, num_docs, clusters, metric, target_num_clusters, merges_per_batch, use_disjunction, cache_lsh, stable, cache_scores, verbose):
    """ Performs LSH-based hierarchical agglomerative clustering (HAC)

        Does specified number of merges per iteration and uses caching if flag is set to True
        Has option to do stable sorting (which takes a little longer)
    """
    def criterion(stat):
        """ Returns a function that checks if LSH buckets meet a particular
            criterion based on a statistic of the bucket distribution

            stat is the name of the statistic desired
        """
        # Criteria for LSH buckets
        min_mean_bucket_size = 2.5
        min_median_bucket_size = 4.4
        min_pairwise_comparisons = 10000

        def mean_criterion(mean, median, num_pairs):
            return mean >= min_mean_bucket_size
        def median_criterion(mean, median, num_pairs):
            return median >= min_median_bucket_size
        def num_comparisons_criterion(mean, median, num_pairs):
            return num_pairs >= min_pairwise_comparisons

        if stat == 'mean':
            return mean_criterion
        elif stat == 'median':
            return median_criterion
        elif stat == 'num_pairs':
            return num_comparisons_criterion
        else:
            print "Error: no known criterion specified"

    # Set criterion function
    bucket_criterion_function = criterion('num_pairs')

    # To start, number of buckets approximately equal to number of documents
    max_bits = int(np.ceil(np.log2(num_docs)))

    # Generate random projection matrix
    proj_mat = np.random.rand(max_bits, num_docs) - 0.5

    # Calculate buckets corresponding to maximum number of hash bits
    if cache_lsh:
        cid_to_hash_cached, buckets_cached = cache_buckets(max_bits, proj_mat, clusters, doc_id_vecs, use_disjunction)
        cid_to_hash = cid_to_hash_cached[max_bits]
        buckets = buckets_cached[max_bits]
    else:
        cid_to_hash_cached = None
        buckets_cached = None
        if use_disjunction:
            cid_to_hash, buckets = compute_buckets_disjunction(max_bits, proj_mat, clusters, doc_id_vecs)
        else:
            cid_to_hash, buckets = compute_buckets(max_bits, proj_mat, clusters, doc_id_vecs)

    # Number of hash bits corresponding to variable "buckets"
    num_bits = max_bits

    cluster_ids = set(clusters.keys())
    bucket_ids = set()
    for b in buckets.itervalues():
        bucket_ids.update(b)

    print "Bucket ids is superset of cluster ids = %s" % bucket_ids.issuperset(cluster_ids)
    print "Bucket ids is subset of cluster ids = %s" % bucket_ids.issubset(cluster_ids)

    num_bits, cid_to_hash, buckets = next_buckets(num_bits, proj_mat, clusters, doc_id_vecs, buckets, cid_to_hash, bucket_criterion_function, use_disjunction=use_disjunction, cache_lsh=cache_lsh, buckets_cached=buckets_cached, cid_to_hash_cached=cid_to_hash_cached)

    cluster_ids = set(clusters.keys())
    bucket_ids = set()
    for b in buckets.itervalues():
        bucket_ids.update(b)

    print "Bucket ids is superset of cluster ids = %s" % bucket_ids.issuperset(cluster_ids)
    print "Bucket ids is subset of cluster ids = %s" % bucket_ids.issubset(cluster_ids)

    # Generate initial score table, which we call "candidates"
    candidates = generate_score_table_lsh(pdict, clusters, buckets, metric)
    if verbose: print "Number of candidates = %s" % (len(candidates))

    # Sort score table
    candidates = sort_score_table(candidates, metric, stable)

    # Initialize merge tree, assumes one word per cluster
    merge_tree = {k: v[0] for k,v in clusters.iteritems()}
    # Counter for next cluster id
    id_next = len(clusters)

    iteration = 0
    merges_executed = 0
    while len(clusters) > 1:
        iteration += 1
        # ids from before merges in the upcoming batch
        ids_before_batch = set(clusters.keys())
        # ids corresponding to clusters merged during batch
        ids_merged = set()
        # current index for best score (start at beginning of sorted list)
        cur_idx = 0

        print "Performing batch of merges"
        # Perform specified number of merges per batch
        for k in range(merges_per_batch):

            if verbose: print "Number of clusters = %s, number of candidates = %s" % (len(clusters), len(candidates))

            if len(clusters) == target_num_clusters:
                # Target number of clusters reached, save clusters in a list
                clusters_target = [c for c in clusters.itervalues()]

            if (len(candidates) == 0):
                if verbose: print "Ran out of candidate merges (empty candidate list), breaking out of loop"
                break

            # Extract first candidate merge
            # Look for first valid candidate (doesn't contain any ids_merged)
            found_valid_cand = False
            while not found_valid_cand and cur_idx < len(candidates):
                (cm1, cm2), merge_score = candidates[cur_idx]
                cur_idx += 1
                if not ids_merged.intersection([cm1, cm2]):
                    found_valid_cand = True

            if not found_valid_cand:
                print "Ran out of candidate merges (no valid ids), breaking out of loop"
                break

            print "[iteration = %s, total merges = %s, num of clusters = %s] Merging clusters (%s, %s) with score %s " % (iteration, merges_executed, len(clusters), cm1, cm2, merge_score)
            if verbose: print clusters[cm1], clusters[cm2]

            # Merge top scoring cluster pair
            wlist1 = clusters[cm1]
            wlist2 = clusters[cm2]
            clusters[id_next] = wlist1 + wlist2
            ids_merged.update([cm1, cm2])

            # Delete clusters that were merged
            del clusters[cm1]
            del clusters[cm2]

            # Update hash tables

            if num_bits > 0:
                cid_to_hash, buckets = update_buckets(cm1, cm2, id_next, buckets, cid_to_hash, doc_id_vecs, num_bits, proj_mat, clusters, use_disjunction)

                if cache_lsh:
                    for i in range(1, num_bits):
                        cid_to_hash_cached[i], buckets_cached[i] = update_buckets(cm1, cm2, id_next, buckets_cached[i], cid_to_hash_cached[i], doc_id_vecs, num_bits, proj_mat, clusters, use_disjunction)

            cluster_ids = set(clusters.keys())
            bucket_ids = set()
            for b in buckets.itervalues():
                bucket_ids.update(b)

            print "Bucket ids is superset of cluster ids = %s" % bucket_ids.issuperset(cluster_ids)
            print "Bucket ids is subset of cluster ids = %s" % bucket_ids.issubset(cluster_ids)

            merges_executed += 1

            # Update merge tree
            merge_tree[id_next] = (cm1, cm2)

            # Increment merge_id
            id_next += 1

        # Calculate next set of buckets
        print "Compute next set of candidates"
        prev_num_bits = num_bits
        if num_bits > 0:
            ti_next = time.time()
            num_bits, cid_to_hash, buckets = next_buckets(num_bits, proj_mat, clusters, doc_id_vecs, buckets, cid_to_hash, bucket_criterion_function, use_disjunction=use_disjunction, cache_lsh=cache_lsh, buckets_cached=buckets_cached, cid_to_hash_cached=cid_to_hash_cached)

            tf_next = time.time()
            print "Time to calculate next buckets = %s sec" % (tf_next - ti_next)

        cluster_ids = set(clusters.keys())
        bucket_ids = set()
        for b in buckets.itervalues():
            bucket_ids.update(b)

        print "Bucket ids is superset of cluster ids = %s" % bucket_ids.issuperset(cluster_ids)
        print "Bucket ids is subset of cluster ids = %s" % bucket_ids.issubset(cluster_ids)
        print "bucket ids - cluster ids = %s" % bucket_ids.difference(cluster_ids)

        # Update score table
        if cache_scores and (prev_num_bits == num_bits):
            print "Deleting candidates..."
            # Delete candidates containing the clusters we just merged
            old_cand_size = len(candidates)
            candidates = [c for c in candidates if not ids_merged.intersection(c[0])]

            if verbose: print "Number of deletions = %s" % (old_cand_size - len(candidates))

            # Add new candidates corresponding to newly merged clusters
            print "Adding new candidates..."

            if num_bits > 0:
                # Update score table according to hash table

                # old ids remaining after merges
                ids = set(sorted(clusters.keys()))
                remain_old_ids = ids_before_batch.intersection(ids)
                # new ids before last batch of merges
                new_ids = ids.difference(ids_before_batch)

                total_cluster_size = 0
                num_pmi_scores_computed = 0
                pairs_computed = set()

                ti_add = time.time()
                for cid_new in new_ids:
                    for h in cid_to_hash[cid_new]:
                        for cid in buckets[h]:
                            if ((cid in remain_old_ids) or (cid > cid_new)):
                                if tuple(sorted([cid, cid_new])) not in pairs_computed:
                                    score = score_clusters(pdict, metric, clusters[cid], clusters[cid_new])
                                    # Sort cluster IDs in ascending order
                                    pair = tuple(sorted([cid, cid_new]))
                                    candidates.append((pair, score))
                                    pairs_computed.add(pair)
                                    l1 = len(clusters[cid])
                                    l2 = len(clusters[cid_new])
                                    num_pmi_scores_computed += l1*(l1-1)/2 + l2*(l2-1)/2
                                    total_cluster_size += l1 + l2

                tf_add = time.time()

                print "Finished adding candidates, added %d candidates, computed %d PMI scores" % (len(pairs_computed), num_pmi_scores_computed)
                if len(pairs_computed) > 0:
                    print "Avg cluster size = %s, avg time per PMI score computed = %s microsec" % ((float(total_cluster_size)/(len(pairs_computed)*2)), (tf_add-ti_add)/num_pmi_scores_computed*1e6)

            else:
                # Ignore hash table and update score table according to clusters

                # Add new candidates corresponding to newly merged clusters
                ids = sorted(clusters.keys())
                # old ids remaining after merges
                remain_old_ids = ids_before_batch.intersection(set(ids))
                if remain_old_ids:
                    # Find newest "old" cluster id
                    last_id = max(remain_old_ids)
                    last_idx = ids.index(last_id)
                else:
                    # Case where all the old clusters were merged during
                    # the last iteration (remain_old_ids is empty set)
                    last_idx = -1
                # Calculate new scores for 1) new id - new id, 2) new id - old id
                # where new ids correspond to newly merged clusters
                for i in range(last_idx+1, len(ids)):
                    for j in range(i):
                        score = score_clusters(pdict, metric, clusters[ids[j]], clusters[ids[i]])
                        # Sort cluster IDs in ascending order
                        pair = tuple(sorted([ids[j], ids[i]]))
                        candidates.append((pair, score))

        else:
            # No caching, re-generate entire score table
            if num_bits > 0:
                ti_generate = time.time()
                candidates = generate_score_table_lsh(pdict, clusters, buckets, metric)
                tf_generate = time.time()
                print "Time to generate new score table = %s sec" % (tf_generate-ti_generate)
            else:
                candidates = generate_score_table_hac(pdict, clusters, metric)
            if verbose: print "Number of candidates = %s" % (len(candidates))

        # Sort score table
        candidates = sort_score_table(candidates, metric, stable)

    return clusters_target, merge_tree

def hac_merge(pdict, clusters, metric, target_num_clusters, merges_per_batch, stable, cache_scores, verbose):
    """ Performs hierarchical agglomerative clustering (HAC)

        Does specified number of merges per iteration and uses caching if flag is set to True
        Has option to do stable sorting (which takes a little longer)
    """
    # Generate initial score table, which we call "candidates"
    candidates = generate_score_table_hac(pdict, clusters, metric)

    # Sort score table
    candidates = sort_score_table(candidates, metric, stable)

    # Initialize merge tree, assumes one word per cluster
    merge_tree = {k: v[0] for k,v in clusters.iteritems()}
    # Counter for next cluster id
    id_next = len(clusters)

    iteration = 0
    merges_executed = 0
    while len(clusters) > 1:
        iteration += 1
        # ids from before merges in the upcoming batch
        ids_before_batch = set(clusters.keys())
        # ids corresponding to clusters merged during batch
        ids_merged = set()
        # current index for best score (start at beginning of sorted list)
        cur_idx = 0
        # Perform specified number of merges per batch
        for k in range(merges_per_batch):
            if verbose: print "Number of clusters = %s, number of candidates = %s" % (len(clusters), len(candidates))

            if len(clusters) == target_num_clusters:
                # Target number of clusters reached, save clusters
                clusters_target = [c for c in clusters.itervalues()]

            if (len(candidates) == 0):
                if verbose: print "Ran out of candidate merges (empty candidate list), breaking out of loop"
                break

            # Extract first candidate merge
            # Look for first valid candidate (doesn't contain any ids_merged)
            found_valid_cand = False
            while not found_valid_cand and cur_idx < len(candidates):
                (cm1, cm2), merge_score = candidates[cur_idx]
                cur_idx += 1
                if not ids_merged.intersection([cm1, cm2]):
                    found_valid_cand = True

            if not found_valid_cand:
                print "Ran out of candidate merges (no valid ids), breaking out of loop"
                break

            print "[iteration = %s, total merges = %s, num of clusters = %s] Merging clusters (%s, %s) with score %s " % (iteration, merges_executed, len(clusters), cm1, cm2, merge_score)
            if verbose: print clusters[cm1], clusters[cm2]

            # Merge top scoring cluster pair
            wlist1 = clusters[cm1]
            wlist2 = clusters[cm2]
            clusters[id_next] = wlist1 + wlist2
            ids_merged.update([cm1, cm2])

            # Delete clusters that were merged
            del clusters[cm1]
            del clusters[cm2]

            merges_executed += 1

            # Update merge tree
            merge_tree[id_next] = (cm1, cm2)

            # Increment merge_id
            id_next += 1

        # Finish updating score table
        if cache_scores:
            # Delete candidates containing the clusters we just merged
            old_cand_size = len(candidates)
            candidates = [c for c in candidates if not ids_merged.intersection(c[0])]

            if verbose: print "Number of deletions = %s" % (old_cand_size - len(candidates))

            # Add new candidates corresponding to newly merged clusters
            ids = sorted(clusters.keys())
            # old ids remaining after merges
            remain_old_ids = ids_before_batch.intersection(set(ids))
            if remain_old_ids:
                # Find newest "old" cluster id
                last_id = max(remain_old_ids)
                last_idx = ids.index(last_id)
            else:
                # Case where all the old clusters were merged during
                # the last iteration (remain_old_ids is empty set)
                last_idx = -1
            # Calculate new scores for 1) new id - new id, 2) new id - old id
            # where new ids correspond to newly merged clusters
            for i in range(last_idx+1, len(ids)):
                for j in range(i):
                    score = score_clusters(pdict, metric, clusters[ids[j]], clusters[ids[i]])
                    # Sort cluster IDs in ascending order
                    pair = tuple(sorted([ids[j], ids[i]]))
                    candidates.append((pair, score))
        else:
            # No caching, re-generate entire score table
            candidates = generate_score_table_hac(pdict, clusters, metric)

        # Sort score table
        candidates = sort_score_table(candidates, metric, stable)

    return clusters_target, merge_tree

def calculate_clusters(pdict, single_counts, vocab, metric, target_num_clusters, method='hac', stable=True, doc_id_vec=None, num_docs=None, num_freq_words=100, merges_per_batch=1, use_disjunction=False, cache_scores=True, verbose=False):
    """ Calculate target number of clusters using specified metric and greedy approaches

        Options:
        - (Heuristic) Optimization using most frequent words
        - (Heuristic) Do multiple merges per iteration
        - Cache scores for non-merged clusters
    """
    if metric not in VALID_METRICS:
        print 'No known coherence metric specified'
        return

    # Clusters are stored as dictionary with key being id,
    # value being list of vocab words
    clusters = {}

    if method == 'most_freq_words':
        if num_freq_words > len(vocab):
            print "Error: parameter for num_freq_words > vocabulary size"
            return
        freq_sort = sorted(single_counts, key=single_counts.get, reverse=True)
        if freq_sort[num_freq_words-1] < TOP_WORDS_CUTOFF:
            print "Error: some counts for top %s words are lower than cutoff" % num_freq_words
            return
        top_freq_words = freq_sort[:num_freq_words]
        for freq_word in top_freq_words:
            clusters.append([freq_word])
        leftover = []
        for v_word in vocab:
            if v_word not in top_freq_words:
                leftover.append(v_word)
        clusters.append(leftover)
    elif method == 'lsh':
        # Generate initial clusters
        for idx, v_word in enumerate(vocab):
            clusters[idx] = [v_word]
        clusters, merge_tree = lsh_merge(pdict, doc_id_vec, num_docs, clusters, metric, target_num_clusters, merges_per_batch, use_disjunction, cache_lsh=True, stable=True, cache_scores=True, verbose=False)
    elif method == 'hac':
        # Generate initial clusters
        for idx, v_word in enumerate(vocab):
            clusters[idx] = [v_word]
        clusters, merge_tree = hac_merge(pdict, clusters, metric, target_num_clusters, merges_per_batch, stable, cache_scores, verbose)

    return clusters, merge_tree

##### UTILITY METHODS ######

def mean_ppmi_of_word_in_cluster(pdict, target_word, cluster):
    """ Given a target word and its cluster, compute mean PMI for all pairs
        containing the target word and another word in the cluster
    """
    # Copy words in cluster
    cwords_minus_target = cluster[:]
    # Remove target word from this copy
    cwords_minus_target.remove(target_word)
    return mean_ppmi_score(pdict, [target_word], cwords_minus_target)

def num_neg_pmi(pdict):
    """ Count number of negative PMIs in PMI dictionary
    """
    total = 0
    neg_count = 0
    for key1 in pdict:
        for key2 in pdict[key1]:
            # Make sure we don't double count
            if key1 < key2:
                total += 1
                if pdict[key1][key2] < 0:
                    neg_count += 1
    return neg_count, neg_count/total

##### PRINT METHODS #####

def print_clusters(pdict, single_counts, clusters, first_n_words):
    """ Print clusters, showing words in each cluster

        Currently, the clusters are not sorted in any particular order

        Multiple ways to display words in each cluster:
        1. Sort by single count frequency
        2. Sort by mean PMI of each word with respect to other words in cluster
        3. Same as (1) but display only first_n_words
        4. Same as (2) but display only first_n_words

        Finally, return sorted versions of the clusters,
        by single count and PMI
    """
    # Sort clusters so largest clusters come first
    clusters.sort(key=lambda x: len(x), reverse=True)

    clusters_by_count = []
    clusters_by_pmi = []

    print "\nWords sorted by single count frequency"
    for idx, c in enumerate(clusters):
        # Sort words in cluster by single count frequency
        clusters_by_count.append(sorted(c, key=lambda w: single_counts[w], reverse=True))
        print "\nCluster %s (%d words)- " % (idx+1, len(c)),
        for w in clusters_by_count[idx]: print w,

    print "\n\nWords sorted by mean PMI"
    for idx, c in enumerate(clusters):
        # Sort words in cluster by mean PMI
        clusters_by_pmi.append(sorted(c, key=lambda w: mean_ppmi_of_word_in_cluster(pdict, w, c), reverse=True))
        print "\nCluster %s (%d words) - " % (idx+1, len(c)),
        for w in clusters_by_pmi[idx]: print w,

    print "\n\nTop %s words by single count frequency" % first_n_words
    for idx, c in enumerate(clusters_by_count):
        print "\nCluster %s (%d words) - " % (idx+1, len(c)),
        for w in clusters_by_count[idx][:first_n_words]: print w,

    print "\n\nTop %s words in clusters by PMI" % first_n_words
    for idx, c in enumerate(clusters_by_pmi):
        print "\nCluster %s (%d words) - " % (idx+1, len(c)),
        for w in clusters_by_pmi[idx][:first_n_words]: print w,

    print "\n"

    return clusters_by_count, clusters_by_pmi

def print_top_pmi_pairs(pdict, vocab, num):
    """ Print highest PMI scores over all word pairs in the vocabulary
    """
    top_scores = []
    # Compile list of highest PMIs
    for v_word in vocab:
        if pdict.get(v_word):
            max_key = max(pdict[v_word], key=pdict[v_word].get)
            # This makes sure we don't double count pairs
            # (depending on how the PMI table is constructed)
            if v_word <= max_key:
                top_scores.append(((v_word, max_key), pdict[v_word][max_key]))
    # Sort such that highest scores are at beginning of list
    top_scores.sort(key=lambda(x): x[1], reverse=True)
    # Print word pairs with highest PMI
    for i in range(num):
        if i >= len(top_scores):
            break
        print "%s, pmi=%s (single count = %s, pair count = %s)" % (top_scores[i][0], top_scores[i][1], single_counts[top_scores[i][0][0]], pair_counts[top_scores[i][0]])

def print_top_pmi_for_freq_words(pdict, num):
    """ Print most frequent words (i.e. words with highest single counts)
        For each most frequent word, print words with highest PMI

        N.B. This implementation assumes that the PMI table duplicates
             word pairs such that pdict[wi][wj] = pdict[wj][wi]
    """
    single_counts_sort = sorted(single_counts, key=single_counts.get, reverse=True)
    for i in range(num):
        if i >= len(single_counts_sort):
            break
        word = single_counts_sort[i]
        print "%s (%s)" % (word, single_counts[word])
        # This assumes that the PMI table duplicates pairs counts
        x = sorted(pdict[word], key=pdict[word].get, reverse=True)
        print x

def print_docs_for_pair(wi, wj):
    """ Print documents that contain word pair (wi, wj)
        and the number of matching documents
    """
    count = 0
    for d in docs:
        text = d['ctext']
        wi_found = re.search(r'\b' + wi + r'\b', text)
        wj_found = re.search(r'\b' + wj + r'\b', text)
        # Print text of matched documents
        if wi_found and wj_found:
            print "[%s] %s" % (d['doc_id'], text)
            count += 1
    print "Total number of matched docs: %s" % count

def print_tree_html(filename, tree):
    """ Write HTML file corresponding to merge tree
    """
    def print_tree_helper(root, node_id):
        if type(tree[root]) != tuple:
            print >>f, r'<li>',
            w = tree[root]
            f.write(w.encode('utf-8'))
            f.write('\n')
        else:
            print >>f, r'<li>',
            print >>f, node_id
            print >>f, r'<ul>'
            # Print left child
            print_tree_helper(tree[root][0], node_id + '0')
            # Print right child
            print_tree_helper(tree[root][1], node_id + '1')
            print >>f, r'</ul>'

    f = open(filename, 'w')
    print >>f, r'<!DOCTYPE html>'
    print >>f, r'<html>'
    print >>f, r'<head>'
    print >>f, r'<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">'
    print >>f, r'<title>Cluster merge tree</title>'
    print >>f, r'</head>'
    print >>f, r'<body>'
    print >>f, r'<ul>'
    print_tree_helper(max(tree.keys()), '0')
    print >>f, r'</ul>'
    print >>f, r'</body>'
    print >>f, r'</html>'
    f.close()

def generate_word_cluster_map(tree):
    """ Generate mapping from words to clusters

        Returns dictionary where each key is word with value being a list
        of cluster ids
    """
    def map_helper(root, ids):
        if type(tree[root]) != tuple:
            word = tree[root]
            wc_map[word] = ids
        else:
            # Print left child
            map_helper(tree[root][0], ids + [ids[-1] + '0'])
            # Print right child
            map_helper(tree[root][1], ids + [ids[-1] + '1'])

    wc_map = {}
    map_helper(max(tree.keys()), ['0'])

    return wc_map

##### MAIN SCRIPT ######

# Read vocab, co-occurence, score data from file
if args.verbose: print "Read JSON documents file..."

with open(args.input) as input_file:
    data = json.load(input_file)

vocabulary = data['vocab']
doc_single_counts = data['single_counts']
pmi_lookup = data['score_table']
doc_id_sparse = data['doc_id_sparse']
num_docs = data['num_docs']

# Need to convert flattened keys back to tuple keys
doc_pair_counts = {}
for flat_key, value in data['pair_counts'].iteritems():
    tuple_key = tuple(sorted(flat_key.split(',')))
    doc_pair_counts[tuple_key] = value

# Calculate clusters
if args.verbose:
    print "Target number of clusters = %s, using %s merges per iteration" % (args.n_clusters, args.merges_per_batch)
    print "Calculating clusters..."

ti = time.time()

if args.method == 'hac':
    clusters, merge_tree  = calculate_clusters(pmi_lookup, doc_single_counts, vocabulary, args.metric, args.n_clusters, method=args.method, merges_per_batch=args.merges_per_batch, verbose=False)
elif args.method == 'lsh':
    if args.verbose and args.disjunction: print "Using disjunction to calculate features"

    if args.random:
        # Replace real PMI data with random data (for testing)
        random_dict = defaultdict(dict)
        for k1 in pmi_lookup:
            for k2 in pmi_lookup[k1]:
                random_dict[k1][k2] = np.random.rand()*10
        pdict = random_dict
    else:
        pdict = pmi_lookup


    # Convert sparse doc id vectors to dense ones
    doc_id_dense = {}
    for word in doc_id_sparse:
        vec = np.zeros(num_docs, dtype=np.uint8)

        if args.random:
            # Replace real doc id data with random data (for testing)
            vec[np.random.rand(num_docs) < 0.0005] = 1
        else:
            vec[doc_id_sparse[word]] = 1

        doc_id_dense[word] = vec

    clusters, merge_tree  = calculate_clusters(pdict, doc_single_counts, vocabulary, args.metric, args.n_clusters, method=args.method, merges_per_batch=args.merges_per_batch, doc_id_vec=doc_id_dense, num_docs=num_docs, use_disjunction=args.disjunction, verbose=False)

tf = time.time()

if args.verbose: print "\nUsed %s metric, clusters found:" % args.metric

clusters_by_count, clusters_by_pmi = print_clusters(pmi_lookup, doc_single_counts, clusters, args.n_top_words)

if args.verbose: print "Clustering took %s seconds for %d merges per iteration" % (tf-ti, args.merges_per_batch)

# Construct dictionary containing clusters and parameters
results = {}

results['clusters'] = clusters
results['clusters_by_count'] = clusters_by_count
results['clusters_by_pmi'] = clusters_by_pmi
results['metric'] = args.metric
results['n_clusters'] = args.n_clusters
results['merges_per_batch'] = args.merges_per_batch
results['merge_tree'] = merge_tree

word_cluster_map = generate_word_cluster_map(merge_tree)

cluster_ids = set()
for ids in word_cluster_map.itervalues():
    cluster_ids = cluster_ids.union(set(ids))

results['word_cluster_map'] = word_cluster_map
results['cluster_ids'] = sorted(list(cluster_ids))

# Write results dictionary to JSON file
with open(args.output, 'w') as output_file:
    json.dump(results, output_file)

# Print merge tree to HTML
if args.tree_html:
    print_tree_html(args.tree_html, merge_tree)

#print_top_pmi_for_freq_words(pmi_dict, 5)
#print_top_pmi_pairs(pmi_dict, vocabulary, 20)
#print_docs_for_pair('american', 'out')

#print "\nNumber of negative PMIs = %s, fraction of negative PMIs = %s" % num_neg_pmi(pmi_dict)

# Print tree test cases

#test_tree1 = {}
#test_tree1[2] = (0,1)
#test_tree1[0] = 'left'
#test_tree1[1] = 'right'

#test_tree2 = {}

#test_tree2[0] = 'a'
#test_tree2[1] = 'b'
#test_tree2[2] = (0,1)
#test_tree2[3] = 'c'
#test_tree2[4] = 'd'
#test_tree2[5] = (3,4)
#test_tree2[6] = (2,5)

#print_tree_html('test.html', merge_tree)
