# Process and clean MALLET documents

from __future__ import division
import argparse, json
import process_util

# Example usage: python mallet_process.py --input docs.txt --output clean_docs.json 

##### PARSER #####

parser = argparse.ArgumentParser(description='Output cleaned text, given MALLET documents', add_help=False)

required_args = parser.add_argument_group('Required arguments')
optional_args = parser.add_argument_group('Optional arguments')
help_arg = parser.add_argument_group('Help')

required_args.add_argument('--input', required=True, help='Input file containing documents, same format as used by MALLET')
required_args.add_argument('--output', required=True, help='Output JSON file containing cleaned documents')

optional_args.add_argument('--stop', required=False, help='File containing stop words, one word per line')
optional_args.add_argument('--mallet_out', required=False, help='Output text file containing cleaned documents in MALLET format')
optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

##### MAIN SCRIPT #####

if args.verbose:
    if args.stop:
        print "Using stop word file %s" % args.stop
    else:
        print "Using default stop word file"
    print "Processing and cleaning documents from %s..." % args.input

# Erase contents of output MALLET document file
if args.mallet_out:
    open(args.mallet_out, 'w').close()
    if args.verbose:
        print "Writing cleaned documents to MALLET format text file %s" % args.mallet_out

# Extract tokens for each document
docs = [] # data only contains document text

# Open files
input_file = open(args.input, 'r')
if args.mallet_out:
    mallet_output_file = open(args.mallet_out, 'a')

# Clean document text
for idx, line in enumerate(input_file):
    # Assumes each line is in the one-doc-per-line MALLET format
    doc_id, label, text = line.strip().split('\t')
    if args.stop:
        ctokens = process_util.clean_tokenize(text, stop_words=args.stop)
    else:
        ctokens = process_util.clean_tokenize(text)

    docs.append(ctokens)

    if args.mallet_out:
        ctext = ' '.join(ctokens)
        mallet_line = '\t'.join([doc_id, label, ctext])
        print >>mallet_output_file, mallet_line

# Close files
input_file.close()
if args.mallet_out:
    mallet_output_file.close()

if args.verbose:
    print "Writing list of cleaned documents to JSON file %s" % args.output

# Write document tokens to JSON file
with open(args.output, 'w') as output_file:
    json.dump(docs, output_file)
