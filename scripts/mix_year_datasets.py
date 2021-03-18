import argparse, json, os
import numpy as np

SIZE = 3000

parser = argparse.ArgumentParser(description='Splits a dataset into others using years as delimiter.')
parser.add_argument('--datasetName', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--outputPath', type=str, help='path to put the resulting split datasets', required=True)
parser.add_argument('--datasetsFolder', type=str, help='years to use as delimiters while splitting', required=True)
args = parser.parse_args()

docs = []

datasets = os.listdir(args.datasetsFolder)
print(f'Datasets found: {datasets}')

for dataset in datasets:
    print(f'Loading {dataset}...')
    documents = json.load(open(os.path.join(args.datasetsFolder, dataset), 'r'))
    permutation = np.random.permutation(len(documents)).astype(int)
    permutation = list(map(lambda x: int(x), permutation))
    # print(f'Permutation: {permutation}')
    selected_docs = [documents[idx] for idx in permutation[0:SIZE]]
    print(f'Selected docs size: {len(selected_docs)}')
    docs += selected_docs

print(f'Final dataset size: {len(docs)}')
path = os.path.join(args.outputPath, f'{args.datasetName}_[subset].json')
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(docs, open(path, 'w'))
