from gensim.corpora import Dictionary
import numpy as np
import json
import os
import argparse
import re
import math


def create_dictionary(documents):
    """Creates word dictionary for given corpus.

    Parameters:
    
    documents (list of str): set of documents

    Returns:

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset
    """
    dictionary = Dictionary(documents)

    # Uncomment the line below if you want to keep a proportion of the tokens in the dictionary
    dictionary.filter_extremes(no_below=1, no_above=1.0)

    return dictionary


OUTPUT_FOLDER = "octis_datasets"
TSV_EXTENSION = ".tsv"
TXT_EXTENSION = ".txt"
READ_MODE = "r"
WRITE_MODE = "w"


parser = argparse.ArgumentParser(description='Preprocessor destined for OCTIS dataset creation')
parser.add_argument('--dataset_file', type=str, help='dataset file', required=True)
parser.add_argument('--train_fraction', type=float, default=0.7, help='percentage of docs reserved for training', required=True)
parser.add_argument('--test_fraction', type=float, default=0.2, help='percentage of docs reserved for testing', required=True)
parser.add_argument('--val_fraction', type=float, default=0.1, help='percentage of docs reserved for validation', required=True)
args = parser.parse_args()

raw_data = json.load(open(args.dataset_file, READ_MODE))
print(f'Total of original documents: {len(raw_data)}')

dataset_folder = args.dataset_file.split(os.path.sep)[-1].split(".")[0]
output_path = os.path.join(OUTPUT_FOLDER, dataset_folder)
os.makedirs(output_path)

documents = np.array(list(map(lambda x: x["data"].replace("\\", "\/\\"), raw_data)))
no_of_documents = len(documents)
no_of_train_documents = math.floor(no_of_documents * float(args.train_fraction))
no_of_test_documents = math.floor(no_of_documents * float(args.test_fraction))
no_of_val_documents = no_of_documents - no_of_train_documents - no_of_test_documents

documents_indexes = np.random.randint(no_of_documents, size=no_of_documents)

train_documents = documents[documents_indexes[:no_of_train_documents]]
test_documents = documents[documents_indexes[no_of_train_documents:no_of_test_documents]]
val_documents = documents[documents_indexes[no_of_test_documents:no_of_val_documents]]

tsv_mapping = {
    "train": train_documents,
    "test": test_documents,
    "val": val_documents,
}

with open(f'{output_path}/corpus${TSV_EXTENSION}', WRITE_MODE) as file:
    for k, v in tsv_mapping.items():
        file.write(f'{v}\t{k}\n')

dictionary = create_dictionary(documents)

with open(f'{output_path}/vocabulary${TXT_EXTENSION}', WRITE_MODE) as file:
    for v in dictionary.id2token.values():
        file.write(f'{v}\n')
