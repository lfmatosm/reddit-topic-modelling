from utils.metrics import get_coherence_score, get_topic_diversity
from utils.misc import update_progress_bar
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd, os, time, datetime, numpy as np, joblib


OUTPUT_PATH = "results/csv/lda/"
READ_MODE = "r"
OUTPUT_FOLDER = "models/lda/"


def get_model_name(k, a, b):
    return "lda_" +  "_k=" + str(k) + "_a=" + str(a) + "_b=" + str(b)


def get_textual_topics(idx_to_word, topic_word_dist):
    topics = []

    for _, topic in enumerate(topic_word_dist):
        topics.append(list(idx_to_word[topic.argsort()]))
    return topics


def train_lda(dictionary, documents, topics, alpha_values, beta_values):
    """Trains multiple LDA models, given training hyperparameters and number of topics (K).
    Also saves training results to a DataFrame object.

    Parameters:
    
    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset

    documents (list of str): set of documents

    topics (list of int): list of topic (K) numbers to train, e.g. [3, 5, 8, 10, 12]

    alpha_values (list of float): alpha hyperparameters for LDA to use, e.g. [0.1, 0.2, 0.35, 0.5, 0.7]

    beta_values (list of float): beta hyperparameters for LDA to use, e.g. [0.1, 0.2, 0.35, 0.5, 0.7]

    Returns:

    pandas.DataFrame: a DataFrame with the training results, including hyperparameters used for each model and coherence scores.
    """
    df = pd.DataFrame({
        "k": [],
        "alpha": [],
        "beta": [],
        "model": [],
        "c_v": [],
        "u_mass": [],
        "c_uci": [],
        "c_npmi": [],
        "diversity": [],
        "path": []
    })

    joined_documents = [ " ".join(document) for document in documents ]

    vectorizer = CountVectorizer(min_df=0.01, max_df=0.85)

    vectorized_documents = vectorizer.fit_transform(joined_documents)

    vec_docs_path = OUTPUT_FOLDER + "lda_count_vectorized_documents"

    os.makedirs(os.path.dirname(vec_docs_path), exist_ok=True)

    joblib.dump(vectorized_documents, vec_docs_path, compress=6)

    total_iterations = len(topics) * len(alpha_values) * len(beta_values)
    print(f'Total iterations for training: {total_iterations}')

    current_iteration = 0

    update_progress_bar(current_iteration, total_iterations)
    for k in topics:
        for a in alpha_values:
            for b in beta_values:
                lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=a, topic_word_prior=b)

                doc_topic_matrix = lda.fit_transform(vectorized_documents)
                
                topic_word_dist = lda.components_

                idx_to_word = np.array(vectorizer.get_feature_names())

                topics = get_textual_topics(idx_to_word, topic_word_dist)

                path_to_save = OUTPUT_FOLDER + get_model_name(k, a, b)

                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

                model = {
                    "vectorizer": vectorizer,
                    "vectorized_docs_path": vec_docs_path,
                    "instance": lda,
                    "doc_topic_matrix": doc_topic_matrix,
                    "topic_word_dist": topic_word_dist,
                    "idx_to_word": idx_to_word,
                    "topics": topics
                }

                joblib.dump(model, path_to_save, compress=6)

                df = df.append({
                    "k": k,
                    "alpha": a,
                    "beta": b,
                    "model": get_model_name(k, a, b),
                    "c_v": get_coherence_score(topics, documents, dictionary, "c_v"),
                    "u_mass": get_coherence_score(topics, documents, dictionary, "u_mass"),
                    "c_uci": get_coherence_score(topics, documents, dictionary, "c_uci"),
                    "c_npmi": get_coherence_score(topics, documents, dictionary, "c_npmi"),
                    "diversity": get_topic_diversity(topics),
                    "path": path_to_save
                }, ignore_index=True)

                current_iteration = current_iteration + 1
                update_progress_bar(current_iteration, total_iterations)

    return df


def train_many_lda(documents, dictionary, topics, alpha_values, beta_values):
    """Kickstarts LDA models training and computes total training time.

    Parameters:
    
    documents (list of str): set of documents

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset

    topics (list of int): K values for training, e.g. [3, 5, 8, 10, 12, 15]

    alpha_values (list of float): alpha hyperparameters for LDA to use, e.g. [0.1, 0.2, 0.35, 0.5, 0.7]

    beta_values (list of float): beta hyperparameters for LDA to use, e.g. [0.1, 0.2, 0.35, 0.5, 0.7]

    Returns:

    str: path where training results where saved as a CSV file
    """
    print("\nBeginning training...")

    training_start_time = time.time()

    df = train_lda(dictionary, documents, topics, alpha_values, beta_values)

    training_end_time = time.time()

    total_training_time_in_seconds = training_end_time - training_start_time

    print(f'Total model training time: {str(datetime.timedelta(seconds=total_training_time_in_seconds))}\n')

    print("\nFinishing training...")

    print(df.head())

    print("Saving results CSV...")

    output_filepath = OUTPUT_PATH + "training_results.csv"

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    df.to_csv(output_filepath)

    print("CSV file with results saved to ", output_filepath)

    return output_filepath
