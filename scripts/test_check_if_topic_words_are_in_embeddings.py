import pandas as pd
import os
import joblib
from gensim.models import KeyedVectors

WORKDIR = f'{os.getcwd()}/evaluation/2021-02-23_Nouns_Only_Pt_3_TEST'
MODELS_PATH = f'{WORKDIR}/models'
CSVS_PATH = f'{WORKDIR}/csvs/'
RESOURCES_PATH = f'{WORKDIR}/resources/'
CTM_MODELS_PATH = f'{MODELS_PATH}/ctm/'
ETM_MODELS_PATH = f'{MODELS_PATH}/etm/'
LDA_MODELS_PATH = f'{MODELS_PATH}/lda/'

ctm_results = pd.read_csv(CSVS_PATH + "ctm_combined_results.csv")
lda_results = pd.read_csv(CSVS_PATH + "lda_results.csv")
etm_results = pd.read_csv(CSVS_PATH + "etm_results.csv")

df = pd.concat([ctm_results, etm_results, lda_results], ignore_index=True)
df = df.sort_values(["c_npmi"], ascending=(False))

model_paths = list(map(lambda x: x.replace('training_outputs/', ''), df['path'].tolist()))
print(f'Models paths: {model_paths}')

print('Reading embeddings...')
embeddings = KeyedVectors.load_word2vec_format('skip_s300.txt', binary=False)

for idx, model_path in enumerate(model_paths):
    model = joblib.load(os.path.join(WORKDIR, model_path))
    topics = model['topics']

    # total = len(topics[0]) * len(topics)
    total = 10 * len(topics)
    found = 0

    for jdx, topic in enumerate(topics):
        for word in topic[:10]:
            try:
                vector = embeddings[word]
                found += 1
            except:
                continue
    
    print(f'Model: {model_path} - Found {found}/{total} words in embeddings')
