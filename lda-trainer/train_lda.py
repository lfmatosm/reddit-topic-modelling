from utils.dataset import create_dictionary, create_corpus
from lda.train import train_many_lda
import time, json, datetime, argparse, pandas as pd
from pprint import pprint

parser = argparse.ArgumentParser(description='Trains LDA models with the given corpora of documents.')

parser.add_argument('--dataset', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--field', type=str, help='field of interest', required=True)
parser.add_argument('--topics', nargs='+', help='list of K values', required=True)
parser.add_argument('--alphas', nargs='+', help='list of alpha values to use', required=True)
parser.add_argument('--betas', nargs='+', help='list of beta values to use', required=True)

args = parser.parse_args()

documents_path = args.dataset
topics = list(map(lambda x: int(x), args.topics))
alphas = list(map(lambda x: float(x), args.alphas))
betas = list(map(lambda x: float(x), args.betas))

processing_time_start = time.time()

#############################################################################################################################
#Documents, corpus and dictionary creation
#############################################################################################################################
print("Documents path: ", documents_path)

dataset = json.load(open(documents_path, 'r'))

documents = [ data[args.field] for data in dataset ]

print("Documents loaded")

documents_processing_starting_time = time.time()

dictionary = create_dictionary(documents)

print("Dictionary created")

documents_processing_end_time = time.time()

documents_processing_total_time_in_seconds = documents_processing_end_time - documents_processing_starting_time

print(f'Documents total preprocessing time: {str(datetime.timedelta(seconds=documents_processing_total_time_in_seconds))}\n')

#############################################################################################################################
#Training
#############################################################################################################################
lda_training_start_time = time.time()

print(f'\n\nLDA models training...')
lda_results_filepath = train_many_lda(documents, dictionary, topics, alphas, betas)

lda_training_end_time = time.time()

total_lda_training_time = lda_training_end_time - lda_training_start_time

print(f'Total LDA training time: {str(datetime.timedelta(seconds=total_lda_training_time))}\n')

processing_time_end = time.time()

total_execution_time_in_seconds = processing_time_end - processing_time_start

print(f'Total execution time: {str(datetime.timedelta(seconds=total_execution_time_in_seconds))}\n')
