import argparse
import pandas as pd
import joblib
import json
from utils.metrics import get_coherence_score, get_topic_diversity
from utils.dataframe import create_model_dataframe, insert_line_in_model_dataframe
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from utils import constants

LDA_FOLDER = "lda/"
BASE_MODELS_PATH = constants.MODELS_FOLDER + LDA_FOLDER


def get_model_name_for_k(k):
    return "lda_k" + str(k)


def get_textual_topics(idx_to_word, topic_word_dist):
    topics = []

    for _, topic in enumerate(topic_word_dist):
        topics.append(list(idx_to_word[topic.argsort()][:20]))
    return topics


def get_topic_word_matrix(idx_to_word, topic_word_dist):
    topics = []

    for _, topic in enumerate(topic_word_dist):
        topics.append(list([idx_to_word[idx] for idx, _ in enumerate(topic)]))
    return topics


parser = argparse.ArgumentParser(description='Trains LDA models with the given corpora of documents.')

parser.add_argument('--train_documents', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--test_documents', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--dictionary', type=str, help='word dictionary path', required=True)
parser.add_argument('--useCv', type=bool, default=False, help='wheter to use CountVectorizer or not', required=False)
parser.add_argument('--topics', nargs='+', help='list of K values', required=True)

args = parser.parse_args()

topics = list(map(lambda x: int(x), args.topics))
print(f'LDA training for K = {topics}')

print("Loading documents and dictionary...")
train_documents = json.load(open(args.train_documents, "r"))
test_documents = json.load(open(args.test_documents, "r"))
dictionary = joblib.load(args.dictionary)
print("Documents and dictionary loaded")

df = create_model_dataframe()

vectorizer = CountVectorizer(min_df=0.01, max_df=0.85) if args.useCv else CountVectorizer()

vectorized_documents = vectorizer.fit_transform(train_documents["joined"])

print(f'Resulting vectorized vocabulary has {len(vectorizer.vocabulary_)} tokens, where {len(vectorizer.stop_words_)} stopwords have been removed')

for k in topics:
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method='online',
        n_jobs=-1,
        random_state=0
    )

    doc_topic_dist = lda.fit_transform(vectorized_documents)
    
    # Normalizing components
    topic_word_dist = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

    idx_to_word = np.array(vectorizer.get_feature_names())

    topic_words = get_textual_topics(idx_to_word, topic_word_dist)

    path_to_save = BASE_MODELS_PATH + get_model_name_for_k(k)

    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

    model = {
        "instance": lda,
        "doc_topic_dist": doc_topic_dist,
        "topic_word_dist": topic_word_dist,
        "idx_to_word": idx_to_word,
        "topic_word_matrix": get_topic_word_matrix(idx_to_word, topic_word_dist),
        "topics": topic_words
    }

    joblib.dump(model, path_to_save, compress=8)

    model_name = get_model_name_for_k(k)
    npmi_train = get_coherence_score(topic_words, train_documents["split"], dictionary, "c_npmi")
    npmi_test = get_coherence_score(topic_words, test_documents["split"], dictionary, "c_npmi")
    diversity = get_topic_diversity(topic_words)
    model_path = LDA_FOLDER + get_model_name_for_k(k)

    df = insert_line_in_model_dataframe(
        df,
        k,
        model_name,
        npmi_train,
        npmi_test,
        diversity,
        model_path,
    )

    print(f'LDA model with {k} topics successfully trained')


print(df.head())

print("Saving results CSV...")

output_filepath = constants.CSV_RESULTS_FOLDER + f'lda_results.csv'

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

df.to_csv(output_filepath)

print("CSV file with results saved to ", output_filepath)
