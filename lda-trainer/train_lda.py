from utils.dataset import create_dictionary, create_corpus, load_textual_dataset, load_dataset
from lda.train import train_many_lda
import time, json, datetime, argparse, pandas as pd
from pprint import pprint
from gensim.corpora import Dictionary
import joblib

parser = argparse.ArgumentParser(description='Trains LDA models with the given corpora of documents.')

parser.add_argument('--dataset', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--dictionary', type=str, help='word dictionary path', required=True)
parser.add_argument('--field', default=None, type=str, help='field of interest', required=False)
parser.add_argument('--useGensimImplementation', type=bool, default=False, help='list of beta values to use')
parser.add_argument('--useExternalDatasetForCoherence', type=bool, default=False, help='wheter to use an external dataset to calculate coherence')
parser.add_argument('--externalDataset', type=str, default=None, help='external dataset file')
parser.add_argument('--useCv', type=bool, default=False, help='wheter to use CountVectorizer or not', required=False)
parser.add_argument('--topics', nargs='+', help='list of K values', required=True)
parser.add_argument('--alphas', nargs='+', help='list of alpha values to use', required=False, default=[1])
parser.add_argument('--betas', nargs='+', help='list of beta values to use', required=False, default=[1])

args = parser.parse_args()

documents_path = args.dataset
topics = list(map(lambda x: int(x), args.topics))
alphas = list(map(lambda x: float(x), args.alphas))
betas = list(map(lambda x: float(x), args.betas))

print(f'Passed args - dataset: {args.dataset}\nfield: {args.field}\nuseGensimImplementation: {args.useGensimImplementation}\nuseExternalDatasetForCoherence: {args.useExternalDatasetForCoherence}\nexternalDataset: {args.externalDataset}\nuseCv: {args.useCv}\ntopics: {args.topics}\nalphas: {args.alphas}\nbetas: {args.betas}')

processing_time_start = time.time()

#############################################################################################################################
#Documents, corpus and dictionary creation
#############################################################################################################################
print("Documents path: ", documents_path)

documents = load_dataset(documents_path, args.field)

print("Documents loaded")

documents_processing_starting_time = time.time()

#dictionary = create_dictionary(documents)
print("Loading word dictionary...")
dictionary = joblib.load(args.dictionary)
print("Word dictionary loaded")

print("Dictionary created")

corpus = None
if (args.useGensimImplementation == True):
    corpus = create_corpus(documents, dictionary)

    print("Corpus created")

external_dataset = None
if (args.useExternalDatasetForCoherence == True and args.externalDataset != None):
    external_dataset = load_textual_dataset(args.externalDataset)

documents_processing_end_time = time.time()

documents_processing_total_time_in_seconds = documents_processing_end_time - documents_processing_starting_time

print(f'Documents total preprocessing time: {str(datetime.timedelta(seconds=documents_processing_total_time_in_seconds))}\n')

#############################################################################################################################
#Training
#############################################################################################################################
lda_training_start_time = time.time()

print(f'\n\nLDA models training...')

if (corpus != None):
    lda_results_filepath = train_many_lda(documents, dictionary, topics, alphas, betas, corpus, external_dataset=external_dataset)
else:
    lda_results_filepath = train_many_lda(documents, dictionary, topics, alphas, betas, external_dataset=external_dataset, use_cv=args.useCv)

lda_training_end_time = time.time()

total_lda_training_time = lda_training_end_time - lda_training_start_time

print(f'Total LDA training time: {str(datetime.timedelta(seconds=total_lda_training_time))}\n')

processing_time_end = time.time()

total_execution_time_in_seconds = processing_time_end - processing_time_start

print(f'Total execution time: {str(datetime.timedelta(seconds=total_execution_time_in_seconds))}\n')
