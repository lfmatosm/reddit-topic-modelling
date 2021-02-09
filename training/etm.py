from embedded_topic_model.models.etm import ETM
from utils import constants
from utils.metrics import get_coherence_score, get_topic_diversity
import argparse
import json
import joblib
import os
import pandas as pd


BASE_MODELS_PATH = constants.MODELS_FOLDER + 'etm/'


def get_model_name(k):
    return f'etm_k{k}'


parser = argparse.ArgumentParser(description="Trains ETM models with the given corpora of documents.")
parser.add_argument("--split_documents", type=str, help="original datset path", required=True)
parser.add_argument("--training_dataset", type=str, help="preprocessed training dataset", required=True)
parser.add_argument("--embeddings", type=str, help="path to word2vec embeddings file to use", required=True)
parser.add_argument("--vocabulary", type=str, help="training vocabulary", required=True)
parser.add_argument('--dictionary', type=str, help="corpus dictionary", required=True)
parser.add_argument("--topics", nargs="+", help="list of K values", required=True)
args = parser.parse_args()

topics = list(map(lambda x: int(x), args.topics))
print(f'ETM training for K = {topics}')

print("Loading documents and dictionary...")
split_documents = json.load(open(args.split_documents, "r"))
dictionary = joblib.load(args.dictionary)
print("Documents and dictionary loaded")

print("Loading ETM training resources...")
train_dataset = joblib.load(args.training_dataset)
vocabulary = joblib.load(args.vocabulary)
print("ETM training resources loaded")

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

for k in topics:
    print(f'Starting training for k={k}...')

    etm_instance = ETM(
        vocabulary,
        embeddings=args.embeddings,
        num_topics=k,
        epochs=300,
        debug_mode=True,
    )
    
    etm_instance.fit(train_dataset)

    topic_words = etm_instance.get_topics(20)

    t_w_mtx = etm_instance.get_topic_word_matrix()

    t_w_dist = etm_instance.get_topic_word_dist()

    d_t_dist = etm_instance.get_document_topic_dist()

    path_to_save = BASE_MODELS_PATH + get_model_name(k)
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    joblib.dump({
        "topics": topic_words,
        "topic_word_dist": t_w_dist.numpy(), #Tensor -> numpy
        "doc_topic_dist": d_t_dist.numpy(),  #Tensor -> numpy
        "idx_to_word": vocabulary,
        "topic_word_matrix": t_w_mtx
    }, path_to_save, compress=8)

    df = df.append({
        "k": k,
        "model": get_model_name(k),
        "c_v": get_coherence_score(topic_words, split_documents, dictionary, "c_v"),
        "u_mass": get_coherence_score(topic_words, split_documents, dictionary, "u_mass"),
        "c_uci": get_coherence_score(topic_words, split_documents, dictionary, "c_uci"),
        "c_npmi": get_coherence_score(topic_words, split_documents, dictionary, "c_npmi"),
        "diversity": get_topic_diversity(topic_words),
        "path": path_to_save,
    }, ignore_index=True)

    print(f'ETM model with {k} topics successfully trained')


print(df.head())

print("Saving results CSV...")

output_filepath = constants.CSV_RESULTS_FOLDER + f'etm_results.csv'

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

df.to_csv(output_filepath)

print("Training finished!")
