import numpy as np
from gensim.models import KeyedVectors
import torch
import os
import argparse
import joblib

torch.device('cpu')

parser = argparse.ArgumentParser(description="TESTE")
parser.add_argument("--embedding_path", type=str, help="TODO", required=True)
parser.add_argument("--vocabulary", type=str, help="TODO", required=True)
args = parser.parse_args()

vocabulary = joblib.load(args.vocabulary)
vocabulary_size = len(vocabulary)
print(f'Vocab size = {vocabulary_size}')
emb_size = 300
embedding_path = args.embedding_path

def get_extension(path):
    assert isinstance(path, str), 'path extension is not str'
    filename = path.split(os.path.sep)[-1]
    return filename.split('.')[-1]

def initialize_embeddings(embedding_path):
    print('Reading embedding_path from original word2vec file...')
    vectors = KeyedVectors.load_word2vec_format(
        embedding_path, 
        binary=False if get_extension(embedding_path) == 'txt' else True,
        limit=1000000,
    )

    print('Generating training embeddings...')
    model_embeddings = np.zeros((vocabulary_size, emb_size))

    for i, word in enumerate(vocabulary):
        try:
            model_embeddings[i] = vectors[word]
        except KeyError:
            model_embeddings[i] = np.random.normal(
                scale=0.6, size=(emb_size, ))
    return torch.from_numpy(model_embeddings).to('cpu')

dictionary = initialize_embeddings(embedding_path)
print(f'Dictionary generated: {len(dictionary)}')
