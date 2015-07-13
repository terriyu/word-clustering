# Process and clean 20newsgroups documents

from __future__ import division
import argparse, json, time
import process_util
from sklearn.datasets import fetch_20newsgroups

# Example usage: python newsgroups_process.py --train train.json --test test.json

##### PARSER #####

parser = argparse.ArgumentParser(description='Output cleaned text for 20newsgroups docs from scikit-learn', add_help=False)

required_args = parser.add_argument_group('Required arguments')
optional_args = parser.add_argument_group('Optional arguments')
help_arg = parser.add_argument_group('Help')

required_args.add_argument('--train', required=True, help='Output JSON file containing cleaned documents from training set')
required_args.add_argument('--test', required=True, help='Output JSON file containing cleaned documents from test set')

optional_args.add_argument('--stop', required=False, help='File containing stop words, one word per line')
optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

##### MAIN SCRIPT #####

ti = time.time()

if args.verbose:
    if args.stop:
        print "Using stop word file %s" % args.stop
    else:
        print "Using default stop word file"
    print "Processing and cleaning documents from 20newsgroups..."

# Extract tokens for each document

# data only contains document text
train_docs = []
test_docs =[]

# Get newsgroups data and remove metadata (headers, footers, and quotes)
train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Clean document text
for idx, line in enumerate(train.data):
    text = line.strip()
    if args.stop:
        ctokens = process_util.clean_tokenize(text, stop_words=args.stop)
    else:
        ctokens = process_util.clean_tokenize(text)

    train_docs.append(ctokens)

for idx, line in enumerate(test.data):
    text = line.strip()
    if args.stop:
        ctokens = process_util.clean_tokenize(text, stop_words=args.stop)
    else:
        ctokens = process_util.clean_tokenize(text)

    test_docs.append(ctokens)

if args.verbose:
    print "Writing list of cleaned documents to JSON files %s and %s" % (args.train, args.test)

# Write document tokens to JSON files
with open(args.train, 'w') as output_file:
    json.dump(train_docs, output_file)
with open(args.test, 'w') as output_file:
    json.dump(test_docs, output_file)

tf = time.time()

if args.verbose:
    print "Processing took %s seconds" % (tf-ti)
