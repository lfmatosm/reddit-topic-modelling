import argparse
import pandas as pd
import joblib
import json
from utils.metrics import get_coherence_score, get_topic_diversity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from utils import constants


BASE_MODELS_PATH = constants.MODELS_FOLDER + "lda/"


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


parser = argparse.ArgumentParser(description='Trains LDA models with the given corpora of split_documents.')

parser.add_argument('--split_documents', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--joined_documents', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--dictionary', type=str, help='word dictionary path', required=True)
parser.add_argument('--useCv', type=bool, default=False, help='wheter to use CountVectorizer or not', required=False)
parser.add_argument('--topics', nargs='+', help='list of K values', required=True)

args = parser.parse_args()

topics = list(map(lambda x: int(x), args.topics))
print(f'LDA training for K = {topics}')


print("Loading documents and dictionary...")
split_documents = json.load(open(args.split_documents, "r"))
joined_documents = json.load(open(args.joined_documents, "r"))
dictionary = joblib.load(args.dictionary)
print("Documents and dictionary loaded")


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

vectorizer = CountVectorizer(min_df=0.01, max_df=0.85) if args.useCv else CountVectorizer()

vectorized_documents = vectorizer.fit_transform(joined_documents)

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

    df = df.append({
        "k": k,
        "model": get_model_name_for_k(k),
        "c_v": get_coherence_score(topic_words, split_documents, dictionary, "c_v"),
        "u_mass": get_coherence_score(topic_words, split_documents, dictionary, "u_mass"),
        "c_uci": get_coherence_score(topic_words, split_documents, dictionary, "c_uci"),
        "c_npmi": get_coherence_score(topic_words, split_documents, dictionary, "c_npmi"),
        "diversity": get_topic_diversity(topic_words),
        "path": path_to_save
    }, ignore_index=True)

    print(f'LDA model with {k} topics successfully trained')


print(df.head())

print("Saving results CSV...")

output_filepath = constants.CSV_RESULTS_FOLDER + f'lda_results.csv'

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

df.to_csv(output_filepath)

print("CSV file with results saved to ", output_filepath)
