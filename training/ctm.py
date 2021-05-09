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
import time
import logging


LOG_FOLDER = "pipeline_logs/"

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
parser.add_argument('--dataset_name', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--lang', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument("--train_documents", type=str, help="dataset path. TXT file", required=True)
parser.add_argument("--validation_documents", type=str, help="dataset path. TXT file", required=True)
parser.add_argument("--data_preparation", type=str, help="dataset path. TXT file", required=True)
parser.add_argument("--prepared_training_dataset", type=str, help="dataset path. TXT file", required=True)
parser.add_argument('--dictionary', type=str, help="word dictionary path", required=True)
parser.add_argument("--topics", nargs="+", help="list of K values", required=True)
parser.add_argument("--inference", type=str, help="list of K values", required=False, default="combined")
args = parser.parse_args()

overall_start = time.time()

LOG_FILE = os.path.join(LOG_FOLDER, f'{args.lang}_ctm_training_{args.dataset_name}.txt')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

topics = list(map(lambda x: int(x), args.topics))
logging.info(f'CTM training for K = {topics}')

logging.info("Loading documents and dictionary...")
train_documents = json.load(open(args.train_documents, "r"))
validation_documents = json.load(open(args.validation_documents, "r"))
logging.info(f'train_documents = {len(train_documents["split"])}')
logging.info(f'validation_documents = {len(validation_documents["split"])}')
dictionary = joblib.load(args.dictionary)
logging.info("Documents and dictionary loaded")

logging.info("Loading CTM training resources...")
data_preparation = joblib.load(args.data_preparation)
prepared_training_dataset = joblib.load(args.prepared_training_dataset)
logging.info("CTM training resources loaded")

df = create_model_dataframe()

logging.info(f'Vocab length: {len(data_preparation.vocab)}')

input_size = 768 if args.lang == "en" else 512

for k in topics:
    start = time.time()

    ctm = CombinedTM(input_size=len(data_preparation.vocab), bert_input_size=input_size, n_components=k) \
        if args.inference == "combined" \
            else ZeroShotTM(input_size=len(data_preparation.vocab), bert_input_size=input_size, n_components=k)

    ctm.fit(prepared_training_dataset)

    end = time.time()

    topic_words = ctm.get_topic_lists(20)
    unnormalized_topic_word_dist = ctm.get_topic_word_matrix()

    softmax = torch.nn.Softmax(dim=1)
    #Normalizes the matrix
    topic_word_dist = softmax(torch.from_numpy(unnormalized_topic_word_dist))

    topics_with_word_probs = get_topics_with_word_probabilities(ctm.train_data.idx2token, topic_word_dist)

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
    npmi_valid = get_coherence_score(topic_words, validation_documents["split"], dictionary, "c_npmi")
    diversity = get_topic_diversity(topic_words)
    model_path = CTM_FOLDER + get_model_name(k, args.inference)

    df = insert_line_in_model_dataframe(
        df,
        k,
        model_name,
        npmi_train,
        npmi_valid,
        diversity,
        model_path,
        end-start,
    )

    logging.info(f'CTM model with {k} topics successfully trained')

logging.info(df.head())

logging.info("Saving results CSV...")

output_filepath = constants.CSV_RESULTS_FOLDER + f'ctm_{args.inference}_results.csv'

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

df.to_csv(output_filepath)

logging.info(f'CSV file with results saved to {output_filepath}')

overall_end = time.time()

logging.info(f'Elapsed training time: {overall_end-overall_start}s')
