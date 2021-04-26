from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import argparse
import os
import joblib

parser = argparse.ArgumentParser(description='Postprocessing test')
parser.add_argument('--csvs', nargs='+', help='list CSV files', required=True)
args = parser.parse_args()


for csv in args.csvs:
    print(f'Postprocessing models found in CSV: {csv}...')
    training_results = pd.read_csv(csv)
    models_paths = training_results['path'].tolist()
    models_base_path = os.path.sep.join(csv.split(os.path.sep)[:-2])

    for model_path in models_paths:
        path_to_load = os.path.join(models_base_path, "models", model_path)
        print(f'Loading model at: {path_to_load}')
        model = joblib.load(path_to_load)
        print(f'model - {model.keys()}')
        # for idx, topic in enumerate(model["topics"]):
        #     print(f'Topic: {",".join(topic)} - most similar words: {model["most_similar_words"][idx][:5]}\n')
        del model

print(f'Postprocessing test finished')
