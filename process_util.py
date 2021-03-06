# Module containing methods for cleaning text and
# computing vocabulary, co-occurence counts, and
# PMI score table

from __future__ import division
from collections import defaultdict
import numpy as np
import nltk, re, string

##### GLOBAL CONSTANTS #####

# Default file to use for stop words
# Format is one word per line
STOP_WORD_FILE = 'stop_lists/mallet_stop.txt'

# Punctuation to remove from documents
PUNCT_REMOVE_TOKEN = set(string.punctuation)
PUNCT_REMOVE_REGEX = set([re.compile(re.escape(y)) for y in string.punctuation])

# Miscellaneous words to remove from documents
MISC_REMOVE = set(["``", "..."])

# Frequency cutoff for inclusion in vocabulary
# Single count frequency must be larger than this number
# to be included in the vocabulary
VOCAB_CUTOFF = 10

##### CREATE DEFAULT STOP WORD SET #####

# Default stop words - each element is a string
stop_words_default = set()

with open(STOP_WORD_FILE) as f:
    lines = f.readlines()

for line in lines:
    stop_words_default.add(line.strip())

##### DATA PROCESSING METHODS #####

def clean_tokenize(text, stop_words=stop_words_default):
    """ Clean and tokenize text: remove punctuation, capitalization, stop words, extra spaces

        Uses the NLTK word tokenizer
    """
    # Replace underscore and dash with space
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'-', ' ', text)
    # Remove colon
    text = re.sub(r':', '', text)
    # Remove capitalization
    text = text.lower()
    # Remove any words starting with a number
    text = re.sub(r'\b[0-9](\w*)\b', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [t for t in tokens if ((t not in PUNCT_REMOVE_TOKEN) and (t not in stop_words) and (t not in MISC_REMOVE) and (t[0] != "'"))]
    return tokens

def create_mallet_clusters(filename, num_clusters, vocab):
    """ Create clusters corresponding to MALLET word topic counts file,
        given the number of clusters, also return the list of words in the MALLET clusters

        Only include a word in the MALLET clusters if it is in our PMI vocabulary
    """
    # Words that appear in the MALLET clusters
    cluster_words = []
    # Clusters - each cluster is a list with entries in format (word, counts)
    clusters_with_counts = [None] * num_clusters

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
            if clusters_with_counts[cluster_idx] is None:
                clusters_with_counts[cluster_idx] = [(word, count)]
            else:
                clusters_with_counts[cluster_idx].append((word, count))

    for c in clusters_with_counts:
        c.sort(key=lambda x: x[1], reverse=True)

    # Clusters with words only (sorted in descending count order)
    clusters_words_only = []
    for c in clusters_with_counts:
        clusters_words_only.append([x[0] for x in c])

    return clusters_words_only, clusters_counts, cluster_words

##### SCORING METHODS ######

def pmi_boolbool(single_counts, pair_counts, N_docs, wi, wj, normalized=False):
    """ Calculate PMI of two words wi, wj and normalizes by number of documents

        - Uses logarithm base 2
        - Only checks if a word occurs or not,
          doesn't account for multiple word occurences in a document
        - If any of the counts are zero, PMI is undefined and None is returned
        - If normalized=True, calculate normalized PMI
    """
    # Enforce alphabetical order in pair
    pair = tuple(sorted([wi, wj]))
    Nij = pair_counts.get(pair,0)
    Ni = single_counts.get(wi,0)
    Nj = single_counts.get(wj,0)
    # The single and pair counts must be non-zero
    # otherwise, the PMI is undefined
    if (Nij != 0) and (Ni != 0) and (Nj != 0):
        pmi = np.log2(N_docs*Nij / (Ni*Nj))
        if normalized:
            pmi = pmi / (- np.log2(Nij / N_docs))
        return pmi
    else:
        return None

##### STATS: CO-OCCURENCE COUNTS, PMI, DOC ID VECTORS #####

def precompute_stats(docs, window=None, norm_pmi=False, cutoff=VOCAB_CUTOFF, verbose=True):

    # Set of words in vocabulary
    vocabulary = set()

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

    if verbose:
        print "Computing single counts..."

    for tokens in docs:
        words = list(set(tokens))
        vocabulary |= set(words)

        doc_len = len(words)
        for i in range(doc_len):
            wi = words[i]
            doc_single_counts[wi] += 1

    if verbose:
        print "Removing low frequency words from vocabulary..."

    vocabulary = {w for w in vocabulary if doc_single_counts[w] > cutoff}

    if verbose:
        print "The vocabulary size is %s words" % len(vocabulary)
        if window:
            print "Computing pair counts with window size = %d" % window
        else:
            print "Computing pair counts for documents..."

    # Doc id vectors for each word in vocabulary
    # Uses sparse representation where each value is a list of doc ids
    # where the word occurs
    doc_id_sparse = defaultdict(list)

    if window:
        for doc_id, tokens in enumerate(docs):
            words = list(set(tokens))
            doc_len = len(words)
            for i in range(doc_len):
                wi = words[i]
                if wi in vocabulary:
                    doc_id_sparse[wi].append(doc_id)
                w_min = max(i - window, 0)
                w_max = min(i + window, doc_len)
                for j in range(w_min, w_max):
                    if j != i:
                        wj = words[j]
                        if wi in vocabulary and wj in vocabulary:
                            pair = tuple(sorted([wi, wj]))
                            doc_pair_counts[pair] += 1
    else:
        for doc_id, tokens in enumerate(docs):
            words = list(set(tokens))
            doc_len = len(words)
            for i in range(doc_len):
                wi = words[i]
                if wi in vocabulary:
                    doc_id_sparse[wi].append(doc_id)
                for j in range(i+1,doc_len):
                    wj = words[j]
                    if wi in vocabulary and wj in vocabulary:
                        pair = tuple(sorted([wi, wj]))
                        doc_pair_counts[pair] += 1

    if verbose:
        if norm_pmi:
            print "Calculating normalized PMI..."
        else:
            print "Calculating PMI..."

    # Iterate only through pairs with non-zero counts
    for pair in doc_pair_counts:
        wi, wj = pair
        pmi = pmi_boolbool(doc_single_counts, doc_pair_counts, len(docs), wi, wj, normalized=norm_pmi)
        if pmi is not None:
            pmi_lookup[wi][wj] = pmi
            # Duplicate PMI for reverse ordering of wi, wj for convenience
            pmi_lookup[wj][wi] = pmi

    return vocabulary, doc_single_counts, doc_pair_counts, pmi_lookup, doc_id_sparse
