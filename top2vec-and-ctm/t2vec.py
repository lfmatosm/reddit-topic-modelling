from top2vec import Top2Vec
import json
import argparse
import pandas as pd
import joblib
import os
from training import constants, utils


def get_model_name(k):
    return f't2v_k{k}'


parser = argparse.ArgumentParser(description="Trains Top2Vec models with the given corpora of documents.")
parser.add_argument("--dataset", type=str, help="dataset path. TXT file", required=True)
args = parser.parse_args()

BASE_MODELS_PATH = constants.MODELS_FOLDER + "t2v/"

#string
print("Loading documents...")
documents = [" ".join(data["body"]) for data in json.load(open(args.dataset, "r"))]
documents2 = [data["body"] for data in json.load(open(args.dataset, "r"))]

print("Creating dictionary...")
dictionary = utils.create_dictionary(documents2)

print("Training model...")
model = Top2Vec(documents, workers=3)

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

n_topics = model.get_num_topics()
print(f'No of topics: {n_topics}')
topic_words, word_scores, topic_nums = model.get_topics()
print(topic_words[0][0:10])

print("Saving model...")
path_to_save = BASE_MODELS_PATH + get_model_name(n_topics)
os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

joblib.dump(model, path_to_save, compress=7)

df = df.append({
    "k": n_topics,
    "model": get_model_name(n_topics),
    "c_v": utils.get_coherence_score(topic_words, documents2, dictionary, "c_v"),
    "u_mass": utils.get_coherence_score(topic_words, documents2, dictionary, "u_mass"),
    "c_uci": utils.get_coherence_score(topic_words, documents2, dictionary, "c_uci"),
    "c_npmi": utils.get_coherence_score(topic_words, documents2, dictionary, "c_npmi"),
    "diversity": utils.get_topic_diversity(topic_words),
    "path": path_to_save
}, ignore_index=True)

print(f'Top2Vecmodel with {n_topics} topics successfully trained')

output_filepath = constants.CSV_RESULTS_FOLDER + "t2v_results.csv"
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
df.to_csv(output_filepath)

print("Training finished!")

# words, word_scores = model.similar_words(keywords=["suicidio"], keywords_neg=[], num_words=10)
# for word, score in zip(words, word_scores):
#     print(f"{word} {score}")
