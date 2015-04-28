Explorations in word coherence for topic discovery

Example commands for generating MALLET files

# Import documents into MALLET format
bin/mallet import-file --input ~/coherence/docs/sotu/sotu_1000_cleaned.txt --keep-sequence --output sotu-1000.mallet

# Create word topic counts file
bin/mallet train-topics --input sotu-1000.mallet --num-topics 10 --random-seed 42 --num-iterations 1000 --word-topic-counts-file word-topic-counts-sotu-1000.txt
