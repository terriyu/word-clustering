Explorations in word coherence for topic discovery

Example commands for generating MALLET files

# Import documents into MALLET format
bin/mallet import-file --input ~/coherence/docs/sotu/sotu_1000_cleaned.txt --keep-sequence --output sotu-1000.mallet

# Create word topic counts file
bin/mallet train-topics --input sotu-1000.mallet --num-topics 10 --random-seed 42 --num-iterations 1000 --word-topic-counts-file word-topic-counts-sotu-1000.txt

##### ISSUES #####

Currently, word pairs are duplicated in construction of PMI table
i.e. pmi_dict[wi][wj] = pmi_dict[wj][wi]
 ==> Should we keep it this way?

For any metric that involves averaging over word pairs,
there is the possibility that a word pair has no entry
in the PMI lookup table.
Currently, if that is the case, the score is considered to be zero.
 ==> Possible solution: truncate all negative PMIs to zero 

##### TODO #####

Implement greedy cluster over most frequent words
(heuristic method from Percy Liang's thesis)

Try approximations for disjunction

Add notes about data structure of JSON files
