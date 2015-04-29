# Writes a JSON file containing vocabulary, co-occurence counts, and score table,
# given input JSON file containing documents

from __future__ import division
import argparse, json
import process_util

# Example usage: python write_counts.py --input docs.json --output counts_scores.json

##### PARSER #####

parser = argparse.ArgumentParser(description='Compute vocabulary, co-occurence counts, and score table and write these to JSON file, given input document JSON file', add_help=False)

required_args = parser.add_argument_group('Required arguments')
optional_args = parser.add_argument_group('Optional arguments')
help_arg = parser.add_argument_group('Help')

required_args.add_argument('--input', required=True, help='Input JSON file containing documents')
required_args.add_argument('--output', required=True, help='Output JSON file containing vocabulary, co-occurence counts, and score table')

optional_args.add_argument('--norm', required=False, action='store_true', help='Calculate normalized PMI instead of conventional PMI')
optional_args.add_argument('--window', required=False, default=None, type=int, help='Window size of +/- argument words for pair counts (default=entire document)')
optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

print args

##### MAIN SCRIPT #####

with open(args.input, 'r') as input_file:
    docs = json.load(input_file)

vocab, single_counts, pair_counts, scores = process_util.counts_and_score_table(docs, window=args.window, norm_pmi=args.norm, json_file=args.output, docs_label=args.input, verbose=True)
