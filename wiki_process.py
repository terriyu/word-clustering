# Process and clean Wikipedia documents

from __future__ import division
import numpy as np
import argparse, codecs, json, os, time
import process_util 

# Example usage:
#
# python wiki_process.py --samples 10000
# python wiki_process.py --all_docs

##### GLOBAL CONSTANTS #####

DEFAULT_ROOT_DIR = 'wiki_data/extracted_no_templates/'

DEFAULT_SEED = 42

NUM_FILES_PER_DIR = 100
NUM_LETTERS = 26 # A-Z
FILE_PREFIX = 'wiki_'

##### PARSER #####

parser = argparse.ArgumentParser(description='Perform co-occurence counts on Wikipedia data', add_help=False)

ex_group = parser.add_argument_group(title='Mutually exclusive arguments (one is required)')
ex_args = ex_group.add_mutually_exclusive_group(required=True)

required_args = parser.add_argument_group('Required arguments')

optional_args = parser.add_argument_group('Optional arguments')

help_arg = parser.add_argument_group('Help')

ex_args.add_argument('--samples', type=int, help='Number of document samples')
ex_args.add_argument('--all_docs', action='store_true', help='Use all documents (no sampling)')

required_args.add_argument('--output', required=True, help='Save processed and cleaned documents to JSON using specified filename')

optional_args.add_argument('--seed', required=False, type=int, default=DEFAULT_SEED, help='Integer to use for random seed when sampling')
optional_args.add_argument('--path', required=False, default=DEFAULT_ROOT_DIR, help='Directory containing Wikipedia data')
optional_args.add_argument('--verbose', required=False, action='store_true', help='Verbose mode')

help_arg.add_argument('-h', '--help', action='help')

args = parser.parse_args()

##### FUNCTIONS #####

def convert_sample_to_path(sample, root):
    """ Convert an integer representing a sample to Wikipedia file path
    """
    dir_num = sample // NUM_FILES_PER_DIR
    file_num = sample % NUM_FILES_PER_DIR 
    directory = chr(dir_num // NUM_LETTERS + ord('A')) + chr(dir_num % NUM_LETTERS + ord('A'))
    file_name = FILE_PREFIX + '%02d' % file_num

    return os.path.join(root, directory, file_name) 

##### MAIN SCRIPT #####

ti = time.time()

wiki_root = args.path
num_doc_samples = args.samples

# List of data directory names, e.g. AA, AB, etc
dirs = [name for name in os.listdir(wiki_root) if os.path.isdir(os.path.join(wiki_root, name))]
dirs.sort()

# Compute number of document files in directories
total_num_files_in_last_dir = len(os.listdir(os.path.join(wiki_root, dirs[-1])))
total_num_files = (len(dirs)-1)*NUM_FILES_PER_DIR + total_num_files_in_last_dir

# Extract raw documents
raw_docs = []
if args.all_docs:
    # Loop through all files
    for i in range(total_num_files):
        file_path = convert_sample_to_path(i, wiki_root)
        with codecs.open(file_path, 'r', encoding='utf-8') as input_file:
            raw_text = input_file.read()
        raw_docs_from_file = raw_text.split('</doc>')[:-1]
        raw_docs.extend(raw_docs_from_file)
        if args.verbose:
            print 'Extracted %d docs from %s' % (len(raw_docs_from_file), file_path)
else:
    # Set random seed
    np.random.seed(args.seed)

    # Draw file samples corresponding to document files 
    draw = np.arange(total_num_files)
    np.random.shuffle(draw)

    # Pick a conservative number of file samples, so that the
    # number of documents in those files will be at least
    # the number of document samples desired, or greater
    if total_num_files < num_doc_samples:
        num_file_samples = total_num_files 
    else:
        num_file_samples = num_doc_samples

    file_samples = draw[:num_file_samples] 

    # Read raw documents from files corresponding to samples
    s = 0
    while (len(raw_docs) < num_doc_samples) and (s < total_num_files):
        file_path = convert_sample_to_path(file_samples[s], wiki_root)
        with codecs.open(file_path, 'r', encoding='utf-8') as input_file:
            raw_text = input_file.read()
        raw_docs_from_file = [re.sub(r'<doc.*>', '', d) for d in raw_text.split('</doc>')[:-1]]
        raw_docs.extend(raw_docs_from_file)
        if args.verbose:
            print 'Extracted %d docs from %s' % (len(raw_docs_from_file), file_path)
        s += 1 

    # This means the requested number of samples was greater than total number of documents
    if len(raw_docs) < num_doc_samples:
        print "Warning: Was only able to process %d documents" % len(raw_docs)

    raw_docs = raw_docs[:num_doc_samples]

# Clean and tokenize documents 
if args.verbose:
    print "Cleaning and tokenizing documents..."

docs = [process_util.clean_tokenize(d) for d in raw_docs]

tf = time.time()

if args.verbose:
    print "Time to process, clean, and tokenize documents = %s seconds" % (tf-ti)

# Save cleaned documents to JSON file
if args.verbose:
    print "Writing cleaned documents to JSON file %s..." % args.output

ti = time.time()

with open(args.output, 'w') as output_file:
    json.dump(docs, output_file)

tf = time.time()

if args.verbose:
    print "Took %s seconds to write JSON FILE" % (tf-ti)
