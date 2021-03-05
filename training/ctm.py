from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from utils import constants
from utils.metrics import get_coherence_score, get_topic_diversity
from utils.dataframe import create_model_dataframe, insert_line_in_model_dataframe
import argparse
import pandas as pd
import joblib
import os
import torch
import json


CTM_FOLDER = 'ctm/'
BASE_MODELS_PATH = f'{constants.MODELS_FOLDER}{CTM_FOLDER}'


def get_model_name(k, inference):
    return f'ctm_k{k}_{inference}'


def get_topics_with_word_probabilities(idx_to_word, topic_word_dist):
    topics = []
    
    for i in range(len(topic_word_dist)):
        words_distribution = topic_word_dist[i].cpu().numpy()
        top_words_indexes = words_distribution.argsort()[-20:]
        descending_top_words_indexes = top_words_indexes[::-1]
        topic_words = [(words_distribution[idx], idx_to_word[idx]) for idx in descending_top_words_indexes]
        topics.append(topic_words)

    return topics


def get_topic_word_matrix(idx_to_word, topic_word_mtx):
    topics = []
    for i in range(len(topic_word_mtx)):
        words_dists = list(topic_word_mtx[i].cpu().numpy())
        component_words = [idx_to_word[idx]
                            for idx, _ in enumerate(words_dists)]
        topics.append(component_words)
    return topics



parser = argparse.ArgumentParser(description="Trains CTM models with the given corpora of split_documents.")
parser.add_argument("--train_documents", type=str, help="dataset path. TXT file", required=True)
parser.add_argument("--test_documents", type=str, help="dataset path. TXT file", required=True)
parser.add_argument("--data_preparation", type=str, help="dataset path. TXT file", required=True)
parser.add_argument("--prepared_training_dataset", type=str, help="dataset path. TXT file", required=True)
parser.add_argument('--dictionary', type=str, help="word dictionary path", required=True)
parser.add_argument("--topics", nargs="+", help="list of K values", required=True)
parser.add_argument("--inference", type=str, help="list of K values", required=False, default="combined")
args = parser.parse_args()

topics = list(map(lambda x: int(x), args.topics))
print(f'CTM training for K = {topics}')

print("Loading documents and dictionary...")
train_documents = json.load(open(args.train_documents, "r"))
test_documents = json.load(open(args.test_documents, "r"))
print(f'train_documents = {len(train_documents["split"])}')
print(f'test_documents = {len(test_documents["split"])}')
dictionary = joblib.load(args.dictionary)
print("Documents and dictionary loaded")

print("Loading CTM training resources...")
data_preparation = joblib.load(args.data_preparation)
prepared_training_dataset = joblib.load(args.prepared_training_dataset)
print("CTM training resources loaded")

df = create_model_dataframe()

print(f'Vocab length: {len(data_preparation.vocab)}')

for k in topics:
    ctm = CombinedTM(input_size=len(data_preparation.vocab), bert_input_size=512, n_components=k) \
        if args.inference == "combined" \
            else ZeroShotTM(input_size=len(data_preparation.vocab), bert_input_size=768, n_components=k)

    ctm.fit(prepared_training_dataset)

    topic_words = ctm.get_topic_lists(20)
    unnormalized_topic_word_dist = ctm.get_topic_word_matrix()

    softmax = torch.nn.Softmax(dim=1)
    #Normalizes the matrix
    topic_word_dist = softmax(torch.from_numpy(unnormalized_topic_word_dist))

    topics_with_word_probs = get_topics_with_word_probabilities(ctm.train_data.idx2token, topic_word_dist)
    print(f'topics_with_word_probs[0] = {topics_with_word_probs[0]}')

    path_to_save = BASE_MODELS_PATH + get_model_name(k, args.inference)
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

    joblib.dump({
        "topics": topic_words,
        "topics_with_word_probs": topics_with_word_probs,
        "topic_word_dist": topic_word_dist.numpy(), #Tensor -> numpy
        "doc_topic_dist": ctm.get_thetas(prepared_training_dataset),
        "idx_to_word": ctm.train_data.idx2token,
        "topic_word_matrix": get_topic_word_matrix(ctm.train_data.idx2token, topic_word_dist),
    }, path_to_save, compress=8)

    model_name = get_model_name(k, args.inference)
    npmi_train = get_coherence_score(topic_words, train_documents["split"], dictionary, "c_npmi")
    npmi_test = get_coherence_score(topic_words, test_documents["split"], dictionary, "c_npmi")
    diversity = get_topic_diversity(topic_words)
    model_path = CTM_FOLDER + get_model_name(k, args.inference)

    df = insert_line_in_model_dataframe(
        df,
        k,
        model_name,
        npmi_train,
        npmi_test,
        diversity,
        model_path,
    )

    print(f'CTM model with {k} topics successfully trained')

print(df.head())

print("Saving results CSV...")

output_filepath = constants.CSV_RESULTS_FOLDER + f'ctm_{args.inference}_results.csv'

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

df.to_csv(output_filepath)

print("Training finished!")
