import numpy as np
import json
import os
import argparse
import re
import math
import functools
import joblib


OUTPUT_FOLDER = "octis/datasets/tsv"
TSV_EXTENSION = "tsv"
TXT_EXTENSION = "txt"
READ_MODE = "r"
WRITE_MODE = "w"


parser = argparse.ArgumentParser(description='Preprocessor destined for OCTIS dataset creation')
parser.add_argument('--dataset_path', type=str, help='base path for the dataset files', required=True)
parser.add_argument('--train_fraction', type=float, default=0.7, help='percentage of docs reserved for training', required=False)
parser.add_argument('--test_fraction', type=float, default=0.2, help='percentage of docs reserved for testing', required=False)
parser.add_argument('--val_fraction', type=float, default=0.1, help='percentage of docs reserved for validation', required=False)
args = parser.parse_args()

dataset_files = os.listdir(args.dataset_path)
raw_documents_files = list(filter(lambda x: re.search(".json", x) is not None, dataset_files))
raw_documents_jsons = [json.load(open(os.path.join(args.dataset_path, dt), READ_MODE)) for dt in raw_documents_files]
raw_documents = list(functools.reduce(lambda x, y: x["joined"] + y["joined"], raw_documents_jsons))

print(f'Total of original documents: {len(raw_documents)}')

dataset_folder = args.dataset_path.split(os.path.sep)[-1]
output_path = os.path.join(OUTPUT_FOLDER, dataset_folder)
os.makedirs(output_path, exist_ok=True)

documents = np.array(raw_documents, dtype=object)
no_of_documents = len(documents)
no_of_train_documents = math.floor(no_of_documents * float(args.train_fraction))
no_of_test_documents = math.floor(no_of_documents * float(args.test_fraction))
no_of_val_documents = no_of_documents - no_of_train_documents - no_of_test_documents

print(f'No of train docs: {no_of_train_documents}')
print(f'No of test docs: {no_of_test_documents}')
print(f'No of val docs: {no_of_val_documents}')

documents_indexes = np.random.randint(no_of_documents, size=no_of_documents)

train_documents = documents[documents_indexes[:no_of_train_documents]]
test_documents = documents[documents_indexes[no_of_train_documents:no_of_train_documents+no_of_test_documents]]
val_documents = documents[documents_indexes[no_of_train_documents+no_of_test_documents:no_of_train_documents+no_of_test_documents+no_of_val_documents]]

tsv_mapping = {
    "train": list(train_documents),
    "test": list(test_documents),
    "val": list(val_documents),
}

with open(f'{output_path}/corpus.{TSV_EXTENSION}', WRITE_MODE) as file:
    for k, v in tsv_mapping.items():
        for document in v:
            file.write(f'{document}\t{k}\n')


dictionary_file = list(filter(lambda x: re.search(".gdict", x), dataset_files))[0]
dictionary = joblib.load(os.path.join(args.dataset_path, dictionary_file))

with open(f'{output_path}/vocabulary.{TXT_EXTENSION}', WRITE_MODE) as file:
    for _, v in dictionary.items():
        file.write(f'{v}\n')
