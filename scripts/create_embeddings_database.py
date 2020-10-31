from pymongo import MongoClient, TEXT
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='No description')
parser.add_argument('--embeddings', type=str, help='embeddings txt file', required=True)
parser.add_argument('--port', type=int, help='local mongo instance port', required=True)
args = parser.parse_args()

client = MongoClient('localhost', args.port)
db = client.word_embeddings_300d
collection = db.embeddings

with open(args.embeddings, 'r') as file:
    for line in file:
        data = line.split()
        word = data[0]
        embedding = list(data[1:])

        collection.insert_one({
            'word': word,
            'embedding': embedding
        })

print(f'Inserted documents: {collection.count_documents({})}')
print(f'Creating text index on "word" key...')
collection.create_index([('word', TEXT)])
print(f'Index creation finished.')
