from __future__ import division
from collections import defaultdict
import argparse, re, sys, time
import numpy as np

##### COMMAND LINE ARGUMENTS #####
# Usage: python calculate_pmi.py --doc doc_file --metric metric --n_clusters nc --merges_per_iter m --n_top_words ntw --mallet_file mallet_file
#
# Arguments
#
# --doc - file containing documents
# --metric - metric to use for clustering
#               ('max', 'min', 'mean', 'geometric', 'harmonic', 'disjunction')
# --n_clusters - target number of clusters (optional, default=10)
# --merges_per_iter - number of greedy merges to perform per iteration (optional, default=3)
# --n_top_words - number of top words in each cluster to display (optional, default=20)
# --mallet_file - Mallet word topics count file for evaluation of clusters (optional)

##### ISSUES #####
# Implement command line arguments with argparse?

# Currently, word pairs are duplicated in construction of PMI table
# i.e. pmi_dict[wi][wj] = pmi_dict[wj][wi]
# Should we keep it this way?

# For any metric that involves averaging over word pairs,
# there is the possibility that a word pair has no entry
# in the PMI lookup table.
# Currently, if that is the case, the score is considered to be zero.
# WRONG??

##### TODO #####

# Do pair counts by N-word windows instead of entire document

# Implement greedy cluster over most frequent words
# (heuristic method from Percy Liang's thesis)

##### GLOBAL CONSTANTS #####

# Frequency cutoff for inclusion in vocabulary
# Single count frequency must be larger than this number
# to be included in the vocabulary
VOCAB_CUTOFF = 5

# Frequency cutoff for inclusion in top words
# Single count frequency must be larger than this number
# to be included in the list of top words
TOP_WORDS_CUTOFF = 10

# File to use for stop words
# Format is one word per line
# May want to change this to command line argument later
STOP_WORD_FILE = 'stop_lists/mallet_stop.txt'

# Punctuation to remove from documents
PUNCT_REMOVE = set([r'\.', r'\?', r'\(', r'\)', '\"', '\'', r',', r'!', r':', r';', r'\$'])

# Miscellaneous words to remove from documents
MISC_REMOVE = set(['th'])

# Scoring metrics available
VALID_METRICS = set(['min', 'max', 'mean', 'geometric', 'harmonic', 'disjunction'])

##### GLOBAL VARIABLES #####

# Stop word list - each element is a string
stop_words = set()

# Set of words in vocabulary
vocabulary = set()

# List containing data about documents
# Each element of list is a dictionary containing info for a single document
# e.g. doc = docs[0]
# doc['doc_id'] = '1946_1'
# doc['label'] = 'English'
# doc['text'] = 'To the Congress of the United States:' (original text)
# doc['ctext'] = 'congress united states' (cleaned text)
documents = []

# Document-level counts for single words
# Keys are words, values are integer counts
doc_single_counts = defaultdict(int)

# Document-level counts for pairs of words
# Keys are tuples (word1, word2), values are integer counts
# The tuples are sorted in alphabetical order such that word1 < word2
doc_pair_counts = defaultdict(int)

# Dictionary containing pre-computed PMI for all word pairs in vocabulary
# Key is a word in the vocabulary, value is a dictionary of word-PMI key-value pairs
# e.g. pmi_dict['food']['prices'] = 1.5
pmi_lookup = defaultdict(dict)

##### PARSER #####

parser = argparse.ArgumentParser(description='Cluster documents using PMI-like metrics.', add_help=False)
required_args = parser.add_argument_group('Required arguments')
optional_args = parser.add_argument_group('Optional arguments')
help_arg = parser.add_argument_group('Help')

required_args.add_argument('--doc', required=True, help='File containing documents, same format as used by jsLDA')
required_args.add_argument('--metric', required=True, choices=['max', 'min', 'mean', 'geometric', 'harmonic', 'disjunction'], help='Metric to use for clustering')

optional_args.add_argument('--n_clusters', required=False, default=10, type=int, help='Target number of clusters (default=10)')
optional_args.add_argument('--merges_per_iter', required=False, default=3, type=int, help='Number of greedy merges to perform per iteration (default=3)')
optional_args.add_argument('--n_top_words', required=False, default=20, help='Number of top words in each cluster to display (default=20)')
optional_args.add_argument('--mallet_file', required=False, help='MALLET word topics count file to use for evaluation of clusters')

help_arg.add_argument('-h', '--help', action='help')

##### DATA PROCESSING METHODS #####

def clean(text):
    """ Clean text: remove punctuation, capitalization,
        stop words, extra spaces
    """
    # Remove punctuation
    for symbol in PUNCT_REMOVE:
        text = re.sub(symbol, '', text)
    # Replace underscore with space
    text = re.sub(r'_', ' ', text)
    # Remove capitalization
    text = text.lower()
    # Remove stop words
    for word in stop_words:
        pattern = r'\b' + word + r'\b'
        text = re.sub(pattern, '', text)
    # Remove numbers
    text = re.sub('[0-9]*', '', text)
    # Remove miscellaneous words
    for word in MISC_REMOVE:
        pattern = r'\b' + word + r'\b'
        text = re.sub(pattern, '', text)
    # Remove extra spaces
    return ' '.join(text.split())

##### SCORING METHODS ######

def pmi_boolbool(single_counts, pair_counts, N_docs, wi, wj):
    """ Calculate PMI of two words wi, wj and normalizes by number of documents

        - Uses logarithm base 2
        - Only checks if a word occurs or not,
          doesn't account for multiple word occurences in a document
        - If any of the counts are zero, PMI is undefined and None is returned
    """
    # Enforce alphabetical order in pair
    pair = tuple(sorted([wi, wj]))
    Nij = pair_counts.get(pair,0)
    Ni = single_counts.get(wi,0)
    Nj = single_counts.get(wj,0)
    # The single and pair counts must be non-zero
    # otherwise, the PMI is undefined
    if (Nij != 0) and (Ni != 0) and (Nj != 0):
        return np.log2(N_docs*Nij/(Ni*Nj))
    else:
        return None

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
    """ Calculate mnimum PMI score among all word pairs
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

def score_clusters(pdict, docs, metric, c1, c2):
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
    elif metric == 'disjunction':
        score = disjunction_pmi_score(docs, set(c1), set(c2))
    else:
        print 'No known coherence metric specified'
        return

    return score

def generate_score_table(pdict, docs, clusters, metric):
    """ Generate score table for all pairwise combinations in clusters,
        according to metric

        Score table has form ([i,j], score) where i and j are cluster indices
    """
    cluster_size = len(clusters)
    candidates = []

    # Calculate initial score table
    for i in range(cluster_size):
        for j in range(i+1,cluster_size):
            score = score_clusters(pdict, docs, metric, clusters[i], clusters[j])
            # Note the cluster indices must be in a list (e.g. [i,j])
            # so that they are mutable
            candidates.append([[i, j], score])

    return candidates

def greedy_merge(docs, pdict, clusters, metric, target_num_clusters, merges_per_iter, cache, verbose):
    """ Performs greedy merging of clusters iteratively until target number of clusters is reached

        Does specified number of merges per iteration and uses caching if flag is set to True
    """
    # Generate initial score table
    candidates = generate_score_table(pdict, docs, clusters, metric)

    # Sort score table
    if metric != 'min':
        # Sort so that highest scores are at beginning of list
        candidates.sort(key=lambda(x): x[1], reverse=True)
    else:
        # Sort so that lowest scores are at beginning of list
        candidates.sort(key=lambda(x): x[1], reverse=False)

    iteration = 0
    while len(clusters) > target_num_clusters:
        iteration += 1
        merges_executed = 0
        # Perform specified number of merges per iteration
        for k in range(merges_per_iter):
            if verbose: print "Number of clusters = %s, number of candidates = %s" % (len(clusters), len(candidates))
            if (len(candidates) == 0) or (len(clusters) <= target_num_clusters):
                print "break"
                break

            (cm1, cm2), merge_score = candidates[0]

            print "[iteration = %s] Merging clusters (%s, %s) with score %s " % (iteration, cm1, cm2, merge_score)
            if verbose: print clusters[cm1], clusters[cm2]

            # Merge top scoring cluster pair
            new_cluster = clusters[cm1] + clusters[cm2]
            del clusters[cm1]
            del clusters[cm2-1]
            clusters.append(new_cluster)
            merges_executed += 1

            deletions = 0
            # Delete entries from table, which contain merged clusters
            for idx in xrange(len(candidates) - 1, -1, -1):
                cand = candidates[idx]
                i,j = cand[0]
                if (i == cm1) or (i == cm2) or (j == cm1) or (j == cm2):
                    if verbose: print "(%s,%s)" % (i,j)
                    del candidates[idx]
                    deletions += 1
            if verbose: print "Number of deletions = %s" % deletions

            # Update cluster indexes in score table
            for cand in candidates:
                # Note j is always larger than i
                i, j = cand[0]
                if i > cm1:
                    if i < cm2:
                        i_updated = i - 1
                    else:
                        i_updated = i - 2
                    cand[0][0] = i_updated
                if j > cm1:
                    if j < cm2:
                        j_updated = j - 1
                    else:
                        j_updated = j - 2
                    cand[0][1] = j_updated

        # Finish updating score table
        if cache:
            cluster_size = len(clusters)
            # Add scores for new cluster to score table
            for i in range(cluster_size - merges_executed, cluster_size):
                for j in range(i):
                    # Note the cluster indices must be in a list (e.g. [i,j])
                    # so that they are mutable
                    score = score_clusters(pdict, docs, metric, clusters[j], clusters[i])
                    if verbose: print "Adding (%s, %s)" % (j, i)
                    candidates.append([[j, i], score])
        else:
            # No caching, re-generate entire score table
            candidates = generate_score_table(pdict, docs, clusters, metric)

        # Sort score table
        if metric != 'min':
            # Sort so that highest scores are at beginning of list
            candidates.sort(key=lambda(x): x[1], reverse=True)
        else:
            # Sort so that lowest scores are at beginning of list
            candidates.sort(key=lambda(x): x[1], reverse=False)

    return clusters

def calculate_clusters(docs, pdict, single_counts, vocab, metric, target_num_clusters, use_freq_words=False, num_freq_words=100, merges_per_iter=1, cache=True, verbose=False):
    """ Calculate target number of clusters using specified metric and greedy approaches

        Options:
        - (Heuristic) Optimization using most frequent words
        - (Heuristic) Do multiple merges per iteration
        - Cache scores for non-merged clusters
    """
    if metric not in VALID_METRICS:
        print 'No known coherence metric specified'
        return

    clusters = []

    if use_freq_words:
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
    else:
        for v_word in vocab:
            clusters.append([v_word])
        clusters = greedy_merge(docs, pdict, clusters, metric, target_num_clusters, merges_per_iter, cache, verbose)

    return clusters

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
    """
    clusters_by_count = []
    clusters_by_pmi = []

    print "Words sorted by single count frequency"
    for idx, c in enumerate(clusters):
        # Sort words in cluster by single count frequency
        clusters_by_count.append(sorted(c, key=lambda w: single_counts[w], reverse=True))
        print "Cluster %s - " % (idx+1),
        print clusters_by_count[idx]

    print "Words sorted by mean PMI"
    for idx, c in enumerate(clusters):
        # Sort words in cluster by mean PMI
        clusters_by_pmi.append(sorted(c, key=lambda w: mean_ppmi_of_word_in_cluster(pdict, w, c), reverse=True))
        print "Cluster %s - " % (idx+1),
        print clusters_by_pmi[idx]

    print "Top %s words by single count frequency" % first_n_words
    for idx, c in enumerate(clusters_by_count):
        print "Cluster %s - " % (idx+1),
        print clusters_by_count[idx][:first_n_words]

    print "Top %s words in clusters by PMI" % first_n_words
    for idx, c in enumerate(clusters_by_pmi):
        print "Cluster %s - " % (idx+1),
        print clusters_by_pmi[idx][:first_n_words]

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

    with open(filename) as f:
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
            else:
                clusters[cluster_idx].append(word)

    return clusters, cluster_words

def calculate_VI(clusters1, clusters2):
    """ Calculate variation of information between two sets of clusters
    """
    # Total number of elements in clusters
    n1 = sum([len(c) for c in clusters1])
    n2 = sum([len(c) for c in clusters2])

    if n1 != n2:
        print "Error: Number of elements in clusters do not match"
        return

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

##### MAIN SCRIPT ######

args = parser.parse_args()

print "Reading in stop word list from %s" % STOP_WORD_FILE

with open(STOP_WORD_FILE) as f:
    lines = f.readlines()

for line in lines:
    stop_words.add(line.strip())

print "Processing documents from %s..." % args.doc

with open(args.doc) as f:
    lines = f.readlines()
    num_docs = len(lines)

print "Computing single counts..."

for line in lines:
    doc = {}
    doc['doc_id'], doc['label'], doc['text'] = line.strip().split('\t')
    doc['ctext'] = clean(doc['text'])
    documents.append(doc)

    words = list(set(doc['ctext'].split()))
    vocabulary |= set(words)

    doc_len = len(words)
    for i in range(doc_len):
        wi = words[i]
        doc_single_counts[wi] += 1

print "Removing low frequency words from vocabulary..."

vocabulary = {w for w in vocabulary if doc_single_counts[w] > VOCAB_CUTOFF}

print "The vocabulary size is %s words" % len(vocabulary)

print "Computing pair counts..."

for doc in documents:
    words = list(set(doc['ctext'].split()))
    doc_len = len(words)
    for i in range(doc_len):
        wi = words[i]
        for j in range(i+1,doc_len):
            wj = words[j]
            if wi in vocabulary and wj in vocabulary:
                pair = tuple(sorted([wi, wj]))
                doc_pair_counts[pair] += 1

print "Calculating PMI..."

# Iterate only through pairs with non-zero counts
for pair in doc_pair_counts:
    wi, wj = pair
    pmi = pmi_boolbool(doc_single_counts, doc_pair_counts, num_docs, wi, wj)
    if pmi is not None:
        pmi_lookup[wi][wj] = pmi
        # Duplicate PMI for reverse ordering of wi, wj for convenience
        pmi_lookup[wj][wi] = pmi

print "Target number of clusters = %s, using %s merges per iteration" % (args.n_clusters, args.merges_per_iter)
print "Calculating clusters..."

ti = time.time()
my_clusters = calculate_clusters(documents, pmi_lookup, doc_single_counts, vocabulary, args.metric, args.n_clusters, use_freq_words=False, num_freq_words=500, merges_per_iter=args.merges_per_iter, verbose=False)
tf = time.time()

print "\nUsed %s metric, clusters found:" % args.metric

print_clusters(pmi_lookup, doc_single_counts, my_clusters, args.n_top_words)

print "Clustering took %s seconds" % (tf-ti)

if args.mallet_file:
    mallet_clusters, mallet_words = create_mallet_clusters(args.mallet_file, args.n_clusters, vocabulary)
    vi = calculate_VI(my_clusters, mallet_clusters)
    print "Variation of information distance between our clusters and MALLET = %s" % vi

#print_top_pmi_for_freq_words(pmi_dict, 5)
#print_top_pmi_pairs(pmi_dict, vocabulary, 20)
#print_docs_for_pair('american', 'out')

#print "\nNumber of negative PMIs = %s, fraction of negative PMIs = %s" % num_neg_pmi(pmi_dict)
