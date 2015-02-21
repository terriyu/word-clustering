from __future__ import division
from collections import defaultdict
import sys, math, re

VOCAB_CUTOFF = 5
NUM_CLUSTERS = 10

stop_words = set()
punctuation = set([r'\.', r'\?', r'\(', r'\)', '\"', '\'', ',', '!', ':'])
docs = []
single_counts = defaultdict(int)
pair_counts = defaultdict(int)
vocab = []
pmi_dict = defaultdict(dict)

def clean(text):
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
            
def pmi_boolbool(num_docs, wi, wj):
    pair = tuple(sorted([wi, wj]))
    Nij = pair_counts.get(pair,0)
    Ni = single_counts.get(wi,0)
    Nj = single_counts.get(wj,0)
    if (Nij != 0) and (Ni != 0) and (Nj != 0):
        #print "Nij = %s, Ni = %s, Nj = %s" % (Nij, Ni, Nj)
        return math.log(num_docs*Nij/(Ni*Nj), 2)
    else:
        #print "no pmi: Nij = %s, Ni = %s, Nj = %s" % (Nij, Ni, Nj)
        return None

##### Scoring methods ######

def disjunction_pmi_score(docs, wset1, wset2):
    count_or1 = 0
    count_or2 = 0
    count_and = 0
    for doc in docs:
        doc_words = set(doc['ctext'].split())
        w1_or_doc = (len(wset1.intersection(doc_words)) != 0)
        w2_or_doc = (len(wset2.intersection(doc_words)) != 0)
        if w1_or_doc:  
            count_or1 += 1
        if w2_or_doc:  
            count_or2 += 1
        if w1_or_doc and w2_or_doc:
            count_and += 1
    num_docs = len(docs) 
    if (count_and != 0) and (count_or1 != 0) and (count_or2 != 0):
        if (count_or1 > VOCAB_CUTOFF) and (count_or2 > VOCAB_CUTOFF):
            disj_pmi = math.log(num_docs*count_and/(count_or1*count_or2), 2)
        else:
            disj_pmi = float("-inf")
    else:
        disj_pmi = float("-inf") 
    return disj_pmi 

def max_single_pmi_score(pdict, wlist1, wlist2):
    max_pair = None
    max_pmi = float("-inf") 
    for word1 in wlist1:
       for word2 in wlist2:
            if word1 in pdict and word2 in pdict[word1]:
                if pdict[word1][word2] > max_pmi:
                   max_pair = tuple(sorted([word1, word2]))
                   max_pmi = pdict[max_pair[0]][max_pair[1]]
    return max_pmi

def min_single_pmi_score(pdict, wlist1, wlist2):
    min_pair = None
    min_pmi = float("inf") 
    for word1 in wlist1:
       for word2 in wlist2:
            if word1 in pdict and word2 in pdict[word1]:
                if pdict[word1][word2] < min_pmi:
                   min_pair = tuple(sorted([word1, word2]))
                   min_pmi = pdict[min_pair[0]][min_pair[1]]
    return min_pmi

def mean_pmi_score(pdict, wlist1, wlist2):
    total_pmi = None 
    for word1 in wlist1:
       for word2 in wlist2:
            if word1 in pdict and word2 in pdict[word1]:
                if total_pmi is None:
                    total_pmi = 0
                total_pmi += pdict[word1][word2]
    if total_pmi is not None:
        return total_pmi / (len(wlist1)*len(wlist2))
    else:
        return float("-inf")

def geometric_pmi_score(pdict, wlist1, wlist2):
    product_pmi = None 
    for word1 in wlist1:
        for word2 in wlist2:
            if word1 in pdict and word2 in pdict[word1]:
                if product_pmi is None:
                    product_pmi = 1     
                pmi = pdict[word1][word2] 
                if pmi > 0:
                    product_pmi *= pmi 
                else:
                    product_pmi = float("-inf") 
                    break
        if product_pmi == float("-inf"):
            break
    if product_pmi == float("-inf"):
        return float("-inf")
    elif product_pmi is None:
        return float("-inf")
    else:
        return product_pmi ** (1/len(wlist1)/len(wlist2))

def harmonic_epmi_score(pdict, wlist1, wlist2):
    recip_total_epmi = None 
    for word1 in wlist1:
       for word2 in wlist2:
            if word1 in pdict and word2 in pdict[word1]:
                if recip_total_epmi is None:
                    recip_total_epmi = 0
                recip_total_epmi += 1/(2**pdict[word1][word2])
    if recip_total_epmi is not None:
        return len(wlist1)*len(wlist2)/recip_total_epmi
    else:
        return float("-inf") 

###### Clustering methods ######

def greedy_cluster(docs, cdict, clusters, method):
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
            candidates.append(((i, j), score))         
    if method != 'min':
        candidates.sort(key=lambda(x): x[1], reverse=True)
    else:
        candidates.sort(key=lambda(x): x[1], reverse=False)

    cm1 = candidates[0][0][0]
    cm2 = candidates[0][0][1]
    print "Merging clusters with score %s" % candidates[0][1]
    print clusters[cm1], clusters[cm2]
    new_cluster = clusters[cm1] + clusters[cm2]
    del clusters[cm1]
    del clusters[cm2-1]
    clusters.append(new_cluster)

    return clusters

def calculate_clusters(docs, cdict, vocab, metric):

    clusters = [] 

    for v_word in vocab:
        clusters.append([v_word])

    while len(clusters) > NUM_CLUSTERS:
        clusters = greedy_cluster(docs, cdict, clusters, metric)

    print "\nUsed %s metric, clusters found:" % metric
    for idx, c in enumerate(clusters):
        print "Cluster %s - " % (idx+1),
        print c

    return clusters            

##### Utility functions ######

def num_neg_pmi(pdict):
    total = 0
    neg_count = 0
    for key1 in pdict:
        for key2 in pdict[key1]:
            total += 1
            if pdict[key1][key2] < 0:
                 neg_count += 1
    return neg_count, neg_count/total

   
##### print functions #####

def print_top_pmi_by_word(pdict, num):
    top_scores = [] 
    vocab_size = len(vocab)
    for i in range(vocab_size):
        wi = vocab[i]
        if pmi_dict.get(wi):
            max_key = max(pmi_dict[wi], key=pmi_dict[wi].get)
            if wi <= max_key:
                top_scores.append(((wi, max_key), pmi_dict[wi][max_key]))
    top_scores.sort(key=lambda(x): x[1], reverse=True)
    for i in range(num):
        if i >= len(top_scores):
            break
        print "%s, pmi=%s (single count = %s, pair count = %s)" % (top_scores[i][0], top_scores[i][1], single_counts[top_scores[i][0][0]], pair_counts[top_scores[i][0]])     

def print_top_pmi_by_freq(pdict, num):   
    single_counts_sort = sorted(single_counts, key=single_counts.get, reverse=True)
    for i in range(num):
        if i >= len(single_counts):
            break
        word = single_counts_sort[i]
        print "%s (%s)" % (word, single_counts[word])
        x = sorted(pdict[word], key=pdict[word].get, reverse=True)
        print x 

def print_docs_pair(wi, wj):
    count = 0
    for d in docs:
        text = d['ctext']
        wi_found = re.search(r'\b' + wi + r'\b', text)
        wj_found = re.search(r'\b' + wj + r'\b', text)
        if wi_found and wj_found:
            print "[%s] %s" % (d['doc_id'], text)
            count += 1
    print "Total number of matched docs: %s" % count

##### Main script ######

print "Reading in documents..."

with open('sotu_stop.txt') as f:
    lines = f.readlines()

for line in lines:
    stop_words.add(line.strip())

with open(sys.argv[1]) as f:
    lines = f.readlines()
    num_docs = len(lines)

vocab_set = set()

print "Processing documents..."

for line in lines:
    doc = {}
    doc['doc_id'], doc['tab'], doc['text'] = line.strip().split('\t')
    doc['ctext'] = clean(doc['text'])
    docs.append(doc)

    words = list(set(doc['ctext'].split()))
    vocab_set |= set(words)

    doc_len = len(words)
    for i in range(doc_len):
        wi = words[i]
        single_counts[wi] += 1

vocab = list(vocab_set)

# Remove low frequency words from vocabulary and single counts table
for v_word in vocab:
    if single_counts[v_word] <= VOCAB_CUTOFF:    
        vocab_set.remove(v_word)
        del single_counts[v_word]

vocab = sorted(list(vocab_set))

for doc in docs:
    words = list(set(doc['ctext'].split()))
    doc_len = len(words)
    for i in range(doc_len):
        wi = words[i]
        for j in range(i+1,doc_len):
            wj = words[j]
            if wi in vocab_set and wj in vocab_set:
                pair = tuple(sorted([wi, wj]))
                pair_counts[pair] += 1    

print "Calculating PMI..."

vocab_size = len(vocab)
for i in range(vocab_size):
    wi = vocab[i]
    # iterate only through non-zero count pairs ???
    for j in range(vocab_size):
        wj = vocab[j]
        pmi = pmi_boolbool(num_docs, wi, wj)
        if pmi:
            pmi_dict[wi][wj] = pmi

print "Print results..."

#print_top_pmi_by_freq(pmi_dict,5)
#print_top_pmi_by_word(pmi_dict,20)
#print_docs_pair('american','out')

calculate_clusters(docs, pmi_dict, vocab, sys.argv[2])

print "\nNumber of negative PMIs = %s, fraction of negative PMIs = %s" % num_neg_pmi(pmi_dict)
