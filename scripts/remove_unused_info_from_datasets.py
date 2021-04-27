import json
import argparse
import os


def get_base_path(dataset_path):
    return os.path.sep.join(dataset_path.split(os.path.sep)[:-1])


def get_dataset_name(dataset_path):
    return dataset_path.split(os.path.sep)[-1]


def get_updated_name(dataset_path):
    path = get_base_path(dataset_path)
    name = get_dataset_name(dataset_path)
    return f'{path}/_{name}'


parser = argparse.ArgumentParser(description='Removes unused data from datasets, leaving just submission body content')
parser.add_argument('--datasets', nargs='+', help='list of datasets', required=True)
args = parser.parse_args()

datasets = args.datasets

for dataset in datasets:
    print(f'Processing "{dataset}"...')
    data = json.load(open(dataset, "r"))
    filtered_data = list(map(lambda document: {
        "body": document["body"]
    }, data))
    updated_name = get_updated_name(dataset)
    json.dump(filtered_data, open(updated_name, "w"))

print("Finished")
