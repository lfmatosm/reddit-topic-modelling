from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.evaluation.measures import CoherenceNPMI
from training import constants, utils
import argparse
import pandas as pd
import joblib
import os
import torch
from gensim.corpora import Dictionary
import joblib


def get_model_name(k):
    return f'ctm_k{k}'


parser = argparse.ArgumentParser(description="Trains CTM models with the given corpora of documents.")
parser.add_argument("--dataset", type=str, help="dataset path. TXT file", required=True)
parser.add_argument('--dictionary', type=str, help='word dictionary path', required=True)
parser.add_argument("--topics", nargs="+", help="list of K values", required=True)
parser.add_argument("--inference", type=str, help="list of K values", required=False, default="combined")
args = parser.parse_args()

topics = list(map(lambda x: int(x), args.topics))

BASE_MODELS_PATH = constants.MODELS_FOLDER + f'ctm_{args.inference}/'

print("Preparing vocab...")
handler = TextHandler(args.dataset)
handler.prepare() # create vocabulary and training data

# generate BERT data
print("Generating BERT data...")
training_bert = bert_embeddings_from_file(args.dataset, "distiluse-base-multilingual-cased")
# training_bert = bert_embeddings_from_file(args.dataset, "bert-base-portuguese-cased")

training_dataset = CTMDataset(handler.bow, training_bert, handler.idx2token)

print("Preparing texts...")
texts = []
with open(args.dataset, "r") as fr:
    texts = [doc.split() for doc in fr.read().splitlines()] # load text for PMI
print("Prepared texts")

print("Loading word dictionary...")
dictionary = joblib.load(args.dictionary)
print("Word dictionary loaded")

df = pd.DataFrame({
    "k": [],
    "model": [],
    "c_v": [],
    "u_mass": [],
    "c_uci": [],
    "c_npmi": [],
    "diversity": [],
    "model_npmi": [],
    "path": []
})

print(f'Beginning training for topic models with K = {topics}')
for k in topics:
    ctm = CTM(
        input_size=len(handler.vocab), 
        bert_input_size=512, 
        inference_type=args.inference, 
        n_components=k
    )

    ctm.fit(training_dataset) # run the model

    topics = ctm.get_topic_lists(20)
    topic_word_dist = ctm.get_topic_word_matrix()

    softmax = torch.nn.Softmax(dim=1)

    topic_word_mtx = softmax(torch.from_numpy(topic_word_dist))

    path_to_save = BASE_MODELS_PATH + get_model_name(k)
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    joblib.dump({
        "topics": topics,
        "topic_word_dist": topic_word_mtx, #Normalizes the matrix
        "doc_topic_dist": ctm.get_thetas(training_dataset),
        "idx_to_word": ctm.train_data.idx2token,
        "topic_word_matrix": utils.get_topic_word_matrix(topic_word_mtx, k, ctm.train_data.idx2token)
    }, path_to_save, compress=7)

    df = df.append({
        "k": k,
        "model": get_model_name(k),
        "c_v": utils.get_coherence_score(topics, texts, dictionary, "c_v"),
        "u_mass": utils.get_coherence_score(topics, texts, dictionary, "u_mass"),
        "c_uci": utils.get_coherence_score(topics, texts, dictionary, "c_uci"),
        "c_npmi": utils.get_coherence_score(topics, texts, dictionary, "c_npmi"),
        "diversity": utils.get_topic_diversity(topics),
        "model_npmi": CoherenceNPMI(texts=texts, topics=topics).score(),
        "path": path_to_save,
    }, ignore_index=True)

    print(f'CTM model with {k} topics successfully trained')


output_filepath = constants.CSV_RESULTS_FOLDER + f'ctm_{args.inference}_results.csv'

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

df.to_csv(output_filepath)

print("Training finished!")
