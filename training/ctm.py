from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from utils import constants, ctm_utils
from utils.metrics import get_coherence_score, get_topic_diversity
import argparse
import pandas as pd
import joblib
import os
import torch
import json


BASE_MODELS_PATH = f'{constants.MODELS_FOLDER}ctm/'


def get_model_name(k, inference):
    return f'ctm_k{k}_{inference}'


parser = argparse.ArgumentParser(description="Trains CTM models with the given corpora of split_documents.")
parser.add_argument("--split_documents", type=str, help="dataset path. TXT file", required=True)
parser.add_argument("--data_preparation", type=str, help="dataset path. TXT file", required=True)
parser.add_argument("--prepared_training_dataset", type=str, help="dataset path. TXT file", required=True)
parser.add_argument('--dictionary', type=str, help="word dictionary path", required=True)
parser.add_argument("--topics", nargs="+", help="list of K values", required=True)
parser.add_argument("--inference", type=str, help="list of K values", required=False, default="combined")
args = parser.parse_args()

topics = list(map(lambda x: int(x), args.topics))
print(f'CTM training for K = {topics}')

print("Loading documents and dictionary...")
split_documents = json.load(open(args.split_documents, "r"))
dictionary = joblib.load(args.dictionary)
print("Documents and dictionary loaded")

print("Loading CTM training resources...")
data_preparation = joblib.load(args.data_preparation)
prepared_training_dataset = joblib.load(args.prepared_training_dataset)
print("CTM training resources loaded")

df = pd.DataFrame({
    "k": [],
    "model": [],
    "c_v": [],
    "u_mass": [],
    "c_uci": [],
    "c_npmi": [],
    "diversity": [],
    "path": []
})

print(f'Vocab length: {len(data_preparation.vocab)}')

for k in topics:
    ctm = CombinedTM(input_size=len(data_preparation.vocab), bert_input_size=512, n_components=k) \
        if args.inference == "combined" \
            else ZeroShotTM(input_size=len(data_preparation.vocab), bert_input_size=768, n_components=k)

    ctm.fit(prepared_training_dataset)

    topic_words = ctm.get_topic_lists(20)
    topic_word_dist = ctm.get_topic_word_matrix()

    softmax = torch.nn.Softmax(dim=1)

    topic_word_mtx = softmax(torch.from_numpy(topic_word_dist))

    path_to_save = BASE_MODELS_PATH + get_model_name(k, args.inference)
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    joblib.dump({
        "topics": topic_words,
        "topic_word_dist": topic_word_mtx, #Normalizes the matrix
        "doc_topic_dist": ctm.get_thetas(prepared_training_dataset),
        "idx_to_word": ctm.train_data.idx2token,
        "topic_word_matrix": ctm_utils.get_topic_word_matrix(topic_word_mtx, k, ctm.train_data.idx2token)
    }, path_to_save, compress=8)

    df = df.append({
        "k": k,
        "model": get_model_name(k, args.inference),
        "c_v": get_coherence_score(topic_words, split_documents, dictionary, "c_v"),
        "u_mass": get_coherence_score(topic_words, split_documents, dictionary, "u_mass"),
        "c_uci": get_coherence_score(topic_words, split_documents, dictionary, "c_uci"),
        "c_npmi": get_coherence_score(topic_words, split_documents, dictionary, "c_npmi"),
        "diversity": get_topic_diversity(topic_words),
        "path": path_to_save,
    }, ignore_index=True)

    print(f'CTM model with {k} topics successfully trained')

print(df.head())

print("Saving results CSV...")

output_filepath = constants.CSV_RESULTS_FOLDER + f'ctm_{args.inference}_results.csv'

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

df.to_csv(output_filepath)

print("Training finished!")
