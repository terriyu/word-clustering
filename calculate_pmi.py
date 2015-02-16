from __future__ import division
from collections import defaultdict
import math, re

VOCAB_CUTOFF = 5
NUM_CLUSTERS = 10

stop_words = set()
punctuation = set([r'\.', r'\?', r'\(', r'\)', '\"', '\'', ',', '!', ':'])
docs = []
num_docs = None
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
            
def pmi_boolbool(wi, wj):
    D = num_docs
    pair = tuple(sorted([wi, wj]))
    Nij = pair_counts.get(pair,0)
    Ni = single_counts.get(wi,0)
    Nj = single_counts.get(wj,0)
    if (Nij != 0) and (Ni != 0) and (Nj != 0):
        #print "Nij = %s, Ni = %s, Nj = %s" % (Nij, Ni, Nj)
        return math.log(D*Nij/(Ni*Nj),2)
    else:
        #print "no pmi: Nij = %s, Ni = %s, Nj = %s" % (Nij, Ni, Nj)
        return None

def best_single_pmi(pdict, wlist1, wlist2):
    top_pair = None
    top_pmi = 0 
    for word1 in wlist1:
       for word2 in wlist2:
            if word1 in pdict and word2 in pdict[word1]:
                if pdict[word1][word2] > top_pmi:
                   top_pair = tuple(sorted([word1, word2]))
                   top_pmi = pdict[top_pair[0]][top_pair[1]]
    return top_pmi

def average_pmi(pdict, wlist1, wlist2):
    total_pmi = 0 
    for word1 in wlist1:
       for word2 in wlist2:
            if word1 in pdict and word2 in pdict[word1]:
                total_pmi += pdict[word1][word2]
    return total_pmi / (len(wlist1)*len(wlist2))

def geometric_epmi(pdict, wlist1, wlist2):
    product_epmi = 0 
    for word1 in wlist1:
       for word2 in wlist2:
            if word1 in pdict and word2 in pdict[word1]:
                product_epmi *= 2**pdict[word1][word2]
    return product_epmi ** (1/len(wlist1)/len(wlist2))

def harmonic_epmi(pdict, wlist1, wlist2):
    recip_total_epmi = 0 
    for word1 in wlist1:
       for word2 in wlist2:
            if word1 in pdict and word2 in pdict[word1]:
                recip_total_epmi += 1/(2**pdict[word1][word2])
    if recip_total_epmi != 0:
        return len(wlist1)*len(wlist2)/recip_total_epmi
    else:
        # if the word lists have no word pairs co-recurring, return a default value of zero
        return 0

def greedy_cluster(pdict, clusters):
    cluster_size = len(clusters)
    candidates = []

    for i in range(cluster_size):
        for j in range(i+1,cluster_size):
            #score = best_single_pmi(pdict, clusters[i],clusters[j])
            #score = average_pmi(pdict, clusters[i],clusters[j])
            #score = geometric_epmi(pdict, clusters[i],clusters[j])
            score = harmonic_epmi(pdict, clusters[i],clusters[j])
            candidates.append(((i, j), score))         
    candidates.sort(key=lambda(x): x[1], reverse=True)

    cm1 = candidates[0][0][0]
    cm2 = candidates[0][0][1]
    print "Merging"
    print clusters[cm1], clusters[cm2]
    new_cluster = clusters[cm1] + clusters[cm2]
    del clusters[cm1]
    del clusters[cm2-1]
    clusters.append(new_cluster)

    return clusters
    
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

with open('sotu_micro.txt') as f:
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
        pmi = pmi_boolbool(wi, wj)
        if pmi:
            pmi_dict[wi][wj] = pmi

print "Print results..."

#print_top_pmi_by_freq(pmi_dict,5)
print_top_pmi_by_word(pmi_dict,20)
#print_docs_pair('american','out')

clusters = [] 

for v_word in vocab:
    clusters.append([v_word])

while len(clusters) > NUM_CLUSTERS:
    clusters = greedy_cluster(pmi_dict, clusters)

print clusters
