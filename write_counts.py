# Writes a JSON file containing vocabulary, co-occurence counts, and score table,
# given input JSON file containing documents

from __future__ import division
import argparse, json, time
import process_util

# Example usage: python write_counts.py --input docs.json --output counts_scores.json

##### PARSER #####

parser = argparse.ArgumentParser(description='Compute vocabulary, co-occurence counts, and score table and write these to JSON file, given input document JSON file', add_help=False)

required_args = parser.add_argument_group('Required arguments')
optional_args = parser.add_argument_group('Optional arguments')
help_arg = parser.add_argument_group('Help')

required_args.add_argument('--input', required=True, help='Input JSON file containing documents')
required_args.add_argument('--output', required=True, help='Output JSON file containing vocabulary, co-occurence counts, and score table')

optional_args.add_argument('--cutoff', required=False, type=int, default=process_util.VOCAB_CUTOFF, help='Count cutoff for vocabulary')
optional_args.add_argument('--N', required=False, type=int, help='Only use first N documents')
optional_args.add_argument('--norm', required=False, action='store_true', help='Calculate normalized PMI instead of conventional PMI')
optional_args.add_argument('--window', required=False, default=None, type=int, help='Window size of +/- argument words for pair counts (default=entire document)')
optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

##### MAIN SCRIPT #####

# Load documents
with open(args.input, 'r') as input_file:
    docs = json.load(input_file)

# If flag specified, use first N documents
if args.N:
    docs = docs[:args.N]

if args.verbose:
    print "Using vocab cutoff = %s" % args.cutoff

ti = time.time()

vocab, single_counts, pair_counts, scores, doc_id_vecs = process_util.precompute_stats(docs, window=args.window, norm_pmi=args.norm, cutoff=args.cutoff, verbose=args.verbose)

# Write JSON file
if args.verbose:
    print "Writing vocabulary, counts, score table, doc id vecs to JSON file %s ..." % args.output
# Need to flatten tuple keys, so that pair counts dictionary is serializable
pair_counts_flat_keys = {}
for key, value in pair_counts.iteritems():
    pair_counts_flat_keys[','.join(key)] = value
# Convert vocabulary set to list, so that it is serializable
results_dict = {'docs_label': args.input, 'num_docs': len(docs), 'norm_pmi': args.norm, 'vocab_cutoff': args.cutoff, 'window': args.window, 'vocab': list(vocab), 'single_counts': single_counts, 'pair_counts': pair_counts_flat_keys, 'score_table': scores, 'doc_id_sparse': doc_id_vecs}
# Write dictionary to JSON file
with open(args.output, 'w') as f:
    json.dump(results_dict, f)

tf = time.time()

if args.verbose:
    print "Computing counts and score table + writing JSON output took %s seconds" % (tf-ti)
