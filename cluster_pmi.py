from __future__ import division
import numpy as np
import argparse, json, time

# Example usage:
# python calculate_pmi.py --doc docs.txt --metric 'mean' -n_clusters 5 --merges_per_iter 10 --n_top_words 10 --mallet_file word_topic_counts.txt

##### GLOBAL CONSTANTS #####

# Frequency cutoff for inclusion in top words
# Single count frequency must be larger than this number
# to be included in the list of top words
TOP_WORDS_CUTOFF = 10

# Scoring metrics available
# Note: temporarily take disjunction out of valid metrics,
#       will work on disjunction more later
VALID_METRICS = set(['min', 'max', 'mean', 'geometric', 'harmonic'])

##### PARSER #####

parser = argparse.ArgumentParser(description='Cluster documents using PMI-based metrics.', add_help=False)

required_args = parser.add_argument_group('Required arguments')
optional_args = parser.add_argument_group('Optional arguments')
help_arg = parser.add_argument_group('Help')

required_args.add_argument('--metric', required=True, choices=VALID_METRICS, help='Metric to use for clustering linkage')
required_args.add_argument('--input', required=True, help='Input JSON file containing pre-computed data, including co-occurence counts and score table')
required_args.add_argument('--output', required=True, help='Output JSON file containing clusters')

optional_args.add_argument('--n_clusters', required=False, default=10, type=int, help='Target number of clusters (default=10)')
optional_args.add_argument('--merges_per_iter', required=False, default=10, type=int, help='Number of greedy merges to perform per iteration (default=10)')
optional_args.add_argument('--n_top_words', required=False, default=20, help='Number of top words in each cluster to display (default=20)')

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

def generate_score_table(pdict, clusters, metric):
    """ Generate score table for all pairwise combinations in clusters,
        according to metric

        Score table has form ([i,j], score) where i and j are cluster indices
    """
    cluster_size = len(clusters)
    candidates = []

    # Calculate initial score table
    for i in range(cluster_size):
        for j in range(i+1,cluster_size):
            score = score_clusters(pdict, metric, clusters[i], clusters[j])
            # Note the cluster indices must be in a list (e.g. [i,j])
            # so that they are mutable
            candidates.append([[i, j], score])

    return candidates

def greedy_merge(pdict, clusters, metric, target_num_clusters, merges_per_iter, cache, verbose):
    """ Performs greedy merging of clusters iteratively until target number of clusters is reached

        Does specified number of merges per iteration and uses caching if flag is set to True
    """
    # Generate initial score table
    candidates = generate_score_table(pdict, clusters, metric)

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
                    score = score_clusters(pdict, metric, clusters[j], clusters[i])
                    if verbose: print "Adding (%s, %s)" % (j, i)
                    candidates.append([[j, i], score])
        else:
            # No caching, re-generate entire score table
            candidates = generate_score_table(pdict, clusters, metric)

        # Sort score table
        if metric != 'min':
            # Sort so that highest scores are at beginning of list
            candidates.sort(key=lambda(x): x[1], reverse=True)
        else:
            # Sort so that lowest scores are at beginning of list
            candidates.sort(key=lambda(x): x[1], reverse=False)

    return clusters

def calculate_clusters(pdict, single_counts, vocab, metric, target_num_clusters, use_freq_words=False, num_freq_words=100, merges_per_iter=1, cache=True, verbose=False):
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
        clusters = greedy_merge(pdict, clusters, metric, target_num_clusters, merges_per_iter, cache, verbose)

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

        Finally, return sorted versions of the clusters,
        by single count and PMI
    """
    clusters_by_count = []
    clusters_by_pmi = []

    print "Words sorted by single count frequency"
    for idx, c in enumerate(clusters):
        # Sort words in cluster by single count frequency
        clusters_by_count.append(sorted(c, key=lambda w: single_counts[w], reverse=True))
        print "Cluster %s (%d words)- " % (idx+1, len(c)),
        print clusters_by_count[idx]

    print "Words sorted by mean PMI"
    for idx, c in enumerate(clusters):
        # Sort words in cluster by mean PMI
        clusters_by_pmi.append(sorted(c, key=lambda w: mean_ppmi_of_word_in_cluster(pdict, w, c), reverse=True))
        print "Cluster %s (%d words) - " % (idx+1, len(c)),
        print clusters_by_pmi[idx]

    print "Top %s words by single count frequency" % first_n_words
    for idx, c in enumerate(clusters_by_count):
        print "Cluster %s (%d words) - " % (idx+1, len(c)),
        print clusters_by_count[idx][:first_n_words]

    print "Top %s words in clusters by PMI" % first_n_words
    for idx, c in enumerate(clusters_by_pmi):
        print "Cluster %s (%d words) - " % (idx+1, len(c)),
        print clusters_by_pmi[idx][:first_n_words]

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

##### MAIN SCRIPT ######

# Read vocab, co-occurence, score data from file
print "Read JSON documents file..."

with open(args.input) as input_file:
    data = json.load(input_file)

vocabulary = data['vocab']
doc_single_counts = data['single_counts']
pmi_lookup = data['score_table']

# Need to convert flattened keys back to tuple keys
doc_pair_counts = {}
for flat_key, value in data['pair_counts'].iteritems():
    tuple_key = tuple(sorted(flat_key.split(',')))
    doc_pair_counts[tuple_key] = value

# Calculate clusters
print "Target number of clusters = %s, using %s merges per iteration" % (args.n_clusters, args.merges_per_iter)
print "Calculating clusters..."

ti = time.time()
my_clusters = calculate_clusters(pmi_lookup, doc_single_counts, vocabulary, args.metric, args.n_clusters, use_freq_words=False, num_freq_words=500, merges_per_iter=args.merges_per_iter, verbose=False)
tf = time.time()

print "\nUsed %s metric, clusters found:" % args.metric

clusters_by_count, clusters_by_pmi = print_clusters(pmi_lookup, doc_single_counts, my_clusters, args.n_top_words)

print "Clustering took %s seconds for %d merges per iteration" % (tf-ti, args.merges_per_iter)

# Construct dictionary containing clusters and parameters
results = {}

results['clusters'] = my_clusters
results['clusters_by_count'] = clusters_by_count
results['clusters_by_pmi'] = clusters_by_pmi
results['metric'] = args.metric
results['n_clusters'] = args.n_clusters
results['merges_per_iter'] = args.merges_per_iter

# Write results dictionary to JSON file
with open(args.output, 'w') as output_file:
    json.dump(results, output_file)

#print_top_pmi_for_freq_words(pmi_dict, 5)
#print_top_pmi_pairs(pmi_dict, vocabulary, 20)
#print_docs_for_pair('american', 'out')

#print "\nNumber of negative PMIs = %s, fraction of negative PMIs = %s" % num_neg_pmi(pmi_dict)
