from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import argparse
import os
import joblib
import json
import time
import liwc
import sys
import logging


LOG_FOLDER = "pipeline_logs/"
OUTPUT_FOLDER = "embeddings/liwc/"

parser = argparse.ArgumentParser(description='Postprocessing of trained models')
parser.add_argument('--lang', type=str, help='list CSV files', required=True)
parser.add_argument('--dataset_name', type=str, help='list CSV files', required=True)
parser.add_argument('--embeddings', type=str, help='embeddings path', required=True)
parser.add_argument('--liwc', type=str, help='embeddings path', required=True)
parser.add_argument('--n_dim', type=int, default=300, help='embeddings path', required=False)
args = parser.parse_args()

start = time.time()

LOG_FILE = os.path.join(LOG_FOLDER, f'{args.lang}_w2v_embeddings_for_liwc_words_{args.dataset_name}.txt')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

start = time.time()

logging.info(f'Loading embeddings at {args.embeddings}...')
embeddings = KeyedVectors.load(args.embeddings, mmap='r')
logging.info(f'Loaded embeddings')

logging.info(f'Loading LIWC dictionary at {liwc}')
parse, category_names = liwc.load_token_parser(args.liwc)
logging.info(f'Categories found: {", ".join(category_names)}')
logging.info(f'Loaded LIWC dictionary')

def get_categories_for_word(word):
    return [category for category in parse(word)]

liwc_embeddings = KeyedVectors(args.n_dim)

words = []
weights = []

found = 0
count = 0

for key in embeddings.vocab.keys():
    count += 1
    word_categories = get_categories_for_word(key)
    if len(word_categories) > 0:
        vector = embeddings[key]
        words.append(key)
        weights.append(vector)
        found += 1

if len(words) > 0 and len(weights) > 0:
    liwc_embeddings.add(words, weights)
else:
    del embeddings
    sys.exit("No words found on LIWC dictionary!")

del embeddings
del words
del weights

output_path = os.path.join(OUTPUT_FOLDER, f'{args.lang}_liwc.w2v')
logging.info(f'Saving LIWC embeddings to {output_path}...')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
liwc_embeddings.save(output_path)
del liwc_embeddings
del output_path
logging.info(f'LIWC embeddings saved')

logging.info(f'{found}/{count} words on embeddings were found on LIWC')
del found
del count

end = time.time()

logging.info(f'Elapsed execution time: {end-start}s')
