from __future__ import division
from collections import defaultdict
import sys, math, re

##### COMMAND LINE ARGUMENTS #####
# Usage: python calculate_pmi.py 
#
# sys.argv[0] - 'calculate_pmi.py' (the script)
# sys.argv[1] - file containing documents
# sys.argv[2] - metric to use for clustering
#               ('max', 'min', 'mean', 'geometric', 'harmonic', 'disjunction')

##### TO DO #####
# Implement command line arguments with argparse?
# Intentionally duplicate count pairs when constructing PMI table?

##### GLOBAL CONSTANTS #####

# Frequency cutoff for inclusion in vocabulary 
# Single count frequency must be larger than this number
# to be included in the vocabulary 
VOCAB_CUTOFF = 5

# Frequency cutoff for inclusion in top words
# Single count frequency must be larger than this number
# to be included in the list of top words
TOP_WORDS_CUTOFF = 10

# Number of clusters to generate
# May want to change this to command line argument later
NUM_CLUSTERS = 10

# File to use for stop words
# Format is one word per line
# May want to change this to command line argument later
STOP_WORD_FILE = 'stop_lists/mallet_stop.txt'

##### GLOBAL VARIABLES #####

# Stop word list - each element is a string
stop_words = set()

# Punctuation to remove from documents
punctuation = set([r'\.', r'\?', r'\(', r'\)', '\"', '\'', ',', '!', ':'])

# Single frequency count for words in vocabulary
# Keys are words, values are integer counts 
single_counts = defaultdict(int)

# Pair frequency count for words in vocabulary
# Keys are tuples (word1, word2), values are integer counts 
# The tuples are sorted in alphabetical order such that word1 < word2
pair_counts = defaultdict(int)

# List of words in vocabulary
vocab = []

# List containing data about documents
# Each element of list is a dictionary containing info for a single document 
# e.g. doc = docs[0]
# doc['doc_id'] = '1946_1'
# doc['label'] = 'English'
# doc['text'] = 'To the Congress of the United States:' (original text)
# doc['ctext'] = 'congress united states' (cleaned text) 
docs = []

# Dictionary containing pre-computed PMI for all word pairs in vocabulary
# Key is a word in the vocabulary, value is a dictionary of word-PMI key-value pairs
# e.g. pmi_dict['food']['prices'] = 1.5
pmi_dict = defaultdict(dict)

##### DATA PROCESSING METHODS #####

def clean(text):
    """ Clean text: remove punctuation, capitalization,
        stop words, extra spaces
    """
    # Remove punctuation
    for symbol in punctuation:
        text = re.sub(symbol, '', text)
    # Remove capitalization
    text = text.lower()
    # Remove stop words
    for word in stop_words:
        pattern = r'\b' + word + r'\b'
        text = re.sub(pattern, '', text)
    # Remove extra spaces
    return ' '.join(text.split())
            
##### SCORING METHODS ######

def pmi_boolbool(num_docs, wi, wj):
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
        return math.log(num_docs*Nij/(Ni*Nj), 2)
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
            if pair[0] in pdict and pair[1] in pdict[pair[0]]:
                if pdict[pair[0]][pair[1]] > max_pmi:
                    max_pmi = pdict[pair[0]][pair[1]]
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
            if pair[0] in pdict and pair[1] in pdict[pair[0]]:
                if pdict[pair[0]][pair[1]] < min_pmi:
                   min_pmi = pdict[pair[0]][pair[1]]
    return min_pmi

def mean_pmi_score(pdict, wlist1, wlist2):
    """ Calculate mean PMI over all word pairs in two word lists,
        given pre-computed PMI dictionary

        - If there is no PMI defined for any of the word pairs,
          return -inf
    """
    total_pmi = None 
    for word1 in wlist1:
        for word2 in wlist2:
            # Enforce alphabetical order in pair
            pair = tuple(sorted([word1, word2]))
            if pair[0] in pdict and pair[1] in pdict[pair[0]]:
                if total_pmi is None:
                    total_pmi = 0
                total_pmi += pdict[pair[0]][pair[1]]
    if total_pmi is not None:
        return total_pmi / (len(wlist1)*len(wlist2))
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
            if pair[0] in pdict and pair[1] in pdict[pair[0]]:
                if product_pmi is None:
                    product_pmi = 1     
                pmi = pdict[pair[0]][pair[1]] 
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
            if pair[0] in pdict and pair[1] in pdict[pair[0]]:
                if total_recip_epmi is None:
                    total_recip_epmi = 0
                total_recip_epmi += 1/(2**pdict[pair[0]][pair[1]])
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
            disj_pmi = math.log(num_docs*count_and/(count_or1*count_or2), 2)
        else:
            disj_pmi = float("-inf")
    else:
        # No documents contain any of these words!
        # Set score to -inf
        disj_pmi = float("-inf") 
    return disj_pmi 

###### CLUSTERING METHODS ######

def greedy_cluster_complicated(docs, cdict, clusters, candidates, calculated, method, merges_per_iter):
    cluster_size = len(clusters)

    for i in range(cluster_size):
        for j in range(i+1,cluster_size):
            if (i not in calculated) or (j not in calculated): 
                if method == 'max':
                    score = max_single_pmi_score(cdict, clusters[i], clusters[j])
                elif method == 'min':
                    score = min_single_pmi_score(cdict, clusters[i], clusters[j])
                elif method == 'mean':
                    score = mean_pmi_score(cdict, clusters[i], clusters[j])
                elif method == 'geometric':
                    score = geometric_pmi_score(cdict, clusters[i], clusters[j])
                elif method == 'harmonic':
                    score = harmonic_epmi_score(cdict, clusters[i], clusters[j])
                elif method == 'disjunction':
                    score = disjunction_pmi_score(docs, set(clusters[i]), set(clusters[j]))
                else:
                    print 'No known coherence metric specified'
                    return
                candidates.append([[i, j], score])         

    if method != 'min':
        candidates.sort(key=lambda(x): x[1], reverse=True)
    else:
        candidates.sort(key=lambda(x): x[1], reverse=False)

    for k in range(merges_per_iter):
        if k > len(candidates):
            break

        cm1 = candidates[k][0][0]
        cm2 = candidates[k][0][1]
        merge_score = candidates[k][1]

        # Update cluster indexes in candidates list

        # Delete entries from table, which contain merged clusters 
        for idx in xrange(len(candidates) - 1, -1, -1):
            cand = candidates[idx]
            i = cand[0][0]
            j = cand[0][1]
            if (i == cm1) or (i == cm2) or (j == cm1) or (j == cm2): 
                del candidates[idx]

        for cand in candidates:
            # Note j is always larger than i
            i = cand[0][0]
            j = cand[0][1]
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

        # Merge clusters (do actual merge)
        print "Merging clusters with score %s" % merge_score 
        print clusters[cm1], clusters[cm2]
        new_cluster = clusters[cm1] + clusters[cm2]
        del clusters[cm1]
        del clusters[cm2-1]
        clusters.append(new_cluster)

    return clusters, candidates, merged

def greedy_cluster(docs, cdict, clusters, method, merges_per_iter):
    cluster_size = len(clusters)
    candidates = []

    for i in range(cluster_size):
        for j in range(i+1,cluster_size):
            if method == 'max':
                score = max_single_pmi_score(cdict, clusters[i], clusters[j])
            elif method == 'min':
                score = min_single_pmi_score(cdict, clusters[i], clusters[j])
            elif method == 'mean':
                score = mean_pmi_score(cdict, clusters[i], clusters[j])
            elif method == 'geometric':
                score = geometric_pmi_score(cdict, clusters[i], clusters[j])
            elif method == 'harmonic':
                score = harmonic_epmi_score(cdict, clusters[i], clusters[j])
            elif method == 'disjunction':
                score = disjunction_pmi_score(docs, set(clusters[i]), set(clusters[j]))
            else:
                print 'No known coherence metric specified'
                return
            candidates.append([[i, j], score])         

    if method != 'min':
        candidates.sort(key=lambda(x): x[1], reverse=True)
    else:
        candidates.sort(key=lambda(x): x[1], reverse=False)

    for k in range(merges_per_iter):
        if (len(candidates) == 0):
            break

        cm1 = candidates[0][0][0]
        cm2 = candidates[0][0][1]
        merge_score = candidates[0][1]

        # Merge clusters (do actual merge)
        print "Merging clusters with score %s" % merge_score 
        print clusters[cm1], clusters[cm2]
        new_cluster = clusters[cm1] + clusters[cm2]
        del clusters[cm1]
        del clusters[cm2-1]
        clusters.append(new_cluster)

        # Update cluster indexes in candidates list

        # Delete entries from table, which contain merged clusters 
        for idx in xrange(len(candidates) - 1, -1, -1):
            cand = candidates[idx]
            i = cand[0][0]
            j = cand[0][1]
            if (i == cm1) or (i == cm2) or (j == cm1) or (j == cm2): 
                del candidates[idx]

        for cand in candidates:
            # Note j is always larger than i
            i = cand[0][0]
            j = cand[0][1]
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

    return clusters

def calculate_clusters(docs, cdict, freq_dict, vocab, metric, use_freq_words=False, num_freq_words=100, merges_per_iter=1):

    clusters = [] 

    if use_freq_words:
        if num_freq_words > len(vocab):
            print "Error: parameter for num_freq_words > vocabulary size"
            return    
        freq_sort = sorted(freq_dict, key=freq_dict.get, reverse=True)
        if freq_sort[num_freq_words-1] < TOP_WORDS_CUTOFF:
            print "Error: counts for top %s words by frequency is too low" % num_freq_words
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

    while len(clusters) > NUM_CLUSTERS:
        clusters = greedy_cluster(docs, cdict, clusters, metric, merges_per_iter)

    return clusters            

##### UTILITY METHODS ######

def mean_pmi_of_word_in_cluster(pdict, target_word, cluster):
    """ Given a target word and its cluster, compute mean PMI for all pairs
        containing the target word and another word in the cluster 
    """
    # Copy words in cluster
    cwords_minus_target = cluster[:]
    # Remove target word from this copy
    cwords_minus_target.remove(target_word)
    return mean_pmi_score(pdict, target_word, cwords_minus_target)             

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

def print_clusters(pdict, clusters, first_n_words):
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
        clusters_by_pmi.append(sorted(c, key=lambda w: mean_pmi_of_word_in_cluster(pdict, w, c), reverse=True))
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

def print_top_pmi_pairs(pdict, num):
    """ Print highest PMI scores over all word pairs in the vocabulary
    """
    top_scores = [] 
    # Compile list of highest PMIs
    for i in range(vocab_size):
        wi = vocab[i]
        if pmi_dict.get(wi):
            max_key = max(pmi_dict[wi], key=pmi_dict[wi].get)
            # This makes sure we don't double count pairs
            # (depending on how the PMI table is constructed)
            if wi <= max_key:
                top_scores.append(((wi, max_key), pmi_dict[wi][max_key]))
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

print "Reading in stop word list from %s" % STOP_WORD_FILE

with open(STOP_WORD_FILE) as f:
    lines = f.readlines()

for line in lines:
    stop_words.add(line.strip())

doc_file = sys.argv[1]

print "Processing documents from %s..." % doc_file 

with open(doc_file) as f:
    lines = f.readlines()
    num_docs = len(lines)

vocab_set = set()

print "Computing single counts..."

for line in lines:
    doc = {}
    doc['doc_id'], doc['label'], doc['text'] = line.strip().split('\t')
    doc['ctext'] = clean(doc['text'])
    docs.append(doc)

    words = list(set(doc['ctext'].split()))
    vocab_set |= set(words)

    doc_len = len(words)
    for i in range(doc_len):
        wi = words[i]
        single_counts[wi] += 1

print "Removing low frequency words from vocabulary..."

vocab = list(vocab_set)

vocab[:] = [word for word in vocab if single_counts[word] > VOCAB_CUTOFF]
vocab.sort()

print "Computing pair counts..."

for doc in docs:
    words = list(set(doc['ctext'].split()))
    doc_len = len(words)
    for i in range(doc_len):
        wi = words[i]
        for j in range(i+1,doc_len):
            wj = words[j]
            if wi in vocab and wj in vocab:
                pair = tuple(sorted([wi, wj]))
                pair_counts[pair] += 1    

print "Calculating PMI..."

# Iterate only through pairs with non-zero counts
for pair in pair_counts:
    wi = pair[0]
    wj = pair[1]
    pmi = pmi_boolbool(num_docs, wi, wj) 
    if pmi is not None:
        pmi_dict[wi][wj] = pmi
        # Duplicate PMI for reverse ordering of wi, wj for convenience
        pmi_dict[wj][wi] = pmi

print "Calculating clusters..."

metric = sys.argv[2]
clusters = calculate_clusters(docs, pmi_dict, single_counts, vocab, metric, use_freq_words=False, num_freq_words=500, merges_per_iter=1)

print "\nUsed %s metric, clusters found:" % metric

print_clusters(pmi_dict, clusters, 20)

#print_top_pmi_for_freq_words(pmi_dict,5)
#print_top_pmi_pairs(pmi_dict,20)
#print_docs_for_pair('american','out')

#print "\nNumber of negative PMIs = %s, fraction of negative PMIs = %s" % num_neg_pmi(pmi_dict)
