# Process and clean 20newsgroups documents

from __future__ import division
import argparse, json, time
import process_util
from sklearn.datasets import fetch_20newsgroups

# Example usage: python newsgroups_process.py --subset train --output train.json

##### PARSER #####

parser = argparse.ArgumentParser(description='Output cleaned text for 20newsgroups docs from scikit-learn', add_help=False)

required_args = parser.add_argument_group('Required arguments')
optional_args = parser.add_argument_group('Optional arguments')
help_arg = parser.add_argument_group('Help')

required_args.add_argument('--subset', required=True, choices=['train', 'test', 'all'], help='Subset argument train, test, or all')
required_args.add_argument('--output', required=True, help='Output JSON file containing cleaned documents from training set')

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
docs = []

# Get newsgroups data and remove metadata (headers, footers, and quotes)
newsgroups = fetch_20newsgroups(subset=args.subset, remove=('headers', 'footers', 'quotes'))

# Clean document text
for idx, line in enumerate(newsgroups.data):
    text = line.strip()
    if args.stop:
        ctokens = process_util.clean_tokenize(text, stop_words=args.stop)
    else:
        ctokens = process_util.clean_tokenize(text)

    docs.append(ctokens)

if args.verbose:
    print "Writing list of cleaned documents to JSON file %s" % args.output

# Write document tokens to JSON files
with open(args.output, 'w') as output_file:
    json.dump(docs, output_file)

tf = time.time()

if args.verbose:
    print "Processing took %s seconds" % (tf-ti)
