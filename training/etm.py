from embedded_topic_model.models.etm import ETM
from utils import constants
from utils.metrics import get_coherence_score, get_topic_diversity
from utils.dataframe import create_model_dataframe, insert_line_in_model_dataframe
import argparse
import json
import joblib
import os
import pandas as pd


ETM_FOLDER = 'etm/'
BASE_MODELS_PATH = constants.MODELS_FOLDER + ETM_FOLDER


def get_model_name(k):
    return f'etm_k{k}'


def get_topics_with_word_probabilities(idx_to_word, topic_word_dist):
    topics = []
    
    for i in range(len(topic_word_dist)):
        words_distribution = topic_word_dist[i].cpu().numpy()
        top_words_indexes = words_distribution.argsort()[-20:]
        descending_top_words_indexes = top_words_indexes[::-1]
        topic_words = [(words_distribution[idx], idx_to_word[idx]) for idx in descending_top_words_indexes]
        topics.append(topic_words)

    return topics


parser = argparse.ArgumentParser(description="Trains ETM models with the given corpora of documents.")
parser.add_argument("--train_documents", type=str, help="original dataset path", required=True)
parser.add_argument("--test_documents", type=str, help="original dataset path", required=True)
parser.add_argument("--training_dataset", type=str, help="preprocessed training dataset", required=True)
parser.add_argument("--embeddings", type=str, help="path to word2vec embeddings file to use", required=True)
parser.add_argument("--vocabulary", type=str, help="training vocabulary", required=True)
parser.add_argument('--dictionary', type=str, help="corpus dictionary", required=True)
parser.add_argument("--topics", nargs="+", help="list of K values", required=True)
args = parser.parse_args()

topics = list(map(lambda x: int(x), args.topics))
print(f'ETM training for K = {topics}')

print("Loading documents and dictionary...")
train_documents = json.load(open(args.train_documents, "r"))
test_documents = json.load(open(args.test_documents, "r"))
dictionary = joblib.load(args.dictionary)
print("Documents and dictionary loaded")

print("Loading ETM training resources...")
train_dataset = joblib.load(args.training_dataset)
vocabulary = joblib.load(args.vocabulary)
print("ETM training resources loaded")

df = create_model_dataframe()

for k in topics:
    print(f'Starting training for k={k}...')

    etm_instance = ETM(
        vocabulary,
        embeddings=args.embeddings,
        num_topics=k,
        epochs=300,
        use_c_format_w2vec=True,
        debug_mode=False,
    )
    
    etm_instance.fit(train_dataset)

    topic_words = etm_instance.get_topics(20)

    t_w_mtx = etm_instance.get_topic_word_matrix()

    t_w_dist = etm_instance.get_topic_word_dist()

    d_t_dist = etm_instance.get_document_topic_dist()

    topics_with_word_probs = get_topics_with_word_probabilities(vocabulary, t_w_dist)
    print(f'topics_with_word_probs[0] = {topics_with_word_probs[0]}')

    path_to_save = BASE_MODELS_PATH + get_model_name(k)
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    joblib.dump({
        "topics": topic_words,
        "topics_with_word_probs": topics_with_word_probs,
        "topic_word_dist": t_w_dist.numpy(), #Tensor -> numpy
        "doc_topic_dist": d_t_dist.numpy(),  #Tensor -> numpy
        "idx_to_word": vocabulary,
        "topic_word_matrix": t_w_mtx
    }, path_to_save, compress=8)

    model_name = get_model_name(k)
    npmi_train = get_coherence_score(topic_words, train_documents["split"], dictionary, "c_npmi")
    npmi_test = get_coherence_score(topic_words, test_documents["split"], dictionary, "c_npmi")
    diversity = get_topic_diversity(topic_words)
    model_path = ETM_FOLDER + get_model_name(k)

    df = insert_line_in_model_dataframe(
        df,
        k,
        model_name,
        npmi_train,
        npmi_test,
        diversity,
        model_path,
    )

    print(f'ETM model with {k} topics successfully trained')


print(df.head())

print("Saving results CSV...")

output_filepath = constants.CSV_RESULTS_FOLDER + f'etm_results.csv'

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

df.to_csv(output_filepath)

print("Training finished!")
