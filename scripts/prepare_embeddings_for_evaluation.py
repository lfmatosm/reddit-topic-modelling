import argparse
import os
from gensim.models import KeyedVectors

parser = argparse.ArgumentParser(description='Prepares word2vec embeddings for optimal use on evaluation notebooks.')
parser.add_argument('--embeddings', nargs='+', help='embeddings paths', required=True)
parser.add_argument('--outputFolder', type=str, help='output folder to put optimized embeddings into', required=True)
args = parser.parse_args()

for embedding_path in args.embeddings:
    embeddings = KeyedVectors.load_word2vec_format(
        embedding_path, 
        binary=False
    )
    print(f'Loaded "{embedding_path}" embeddings')
    embedding_name = embedding_path.split(os.path.sep)[-1].replace('.txt', '.w2v')
    path = os.path.join(args.outputFolder, embedding_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    embeddings.save(path)
    del embeddings
    print(f'Saved gensim embeddings to "{path}"')
