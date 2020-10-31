from utils.metrics import get_coherence_score, get_topic_diversity, get_coherence_score_gensim
from utils.misc import update_progress_bar
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd, os, time, datetime, numpy as np, joblib, re
from gensim.models.ldamulticore import LdaMulticore
from training import constants


OUTPUT_PATH = constants.CSV_RESULTS_FOLDER + "lda/"
READ_MODE = "r"
OUTPUT_FOLDER = constants.MODELS_FOLDER + "/lda/"


def get_model_name(k, a, b):
    return "lda_k" + str(k) + "a" + str(a) + "b" + str(b)

def get_model_name_for_k(k):
    return "lda_k" + str(k)


def get_textual_topics(idx_to_word, topic_word_dist):
    topics = []

    for _, topic in enumerate(topic_word_dist):
        topics.append(list(idx_to_word[topic.argsort()][:20]))
    return topics


def get_gensim_topics(model, k):
    tuples = model.print_topics(num_topics=k, num_words=20)
    
    return [list(map(lambda x: x.strip(), re.sub(r'[\d*.\d*]|\*|"', '', t[1]).split('+'))) for t in tuples]


def train_lda_gensim(corpus, dictionary, documents, topics, alpha_values, beta_values):
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
        "gensim_default_coherence": [],
        "gensim_alt_coherence": [],
        "path": []
    })

    total_iterations = len(topics) * len(alpha_values) * len(beta_values)
    print(f'Total iterations for training: {total_iterations}')

    current_iteration = 0
    for k in topics:
        for alpha in alpha_values:
            for beta in beta_values:
                lda_model = LdaMulticore(
                    workers=3, 
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=k,
                    passes=10,
                    per_word_topics=True,
                    alpha=alpha,
                    eta=beta
                )

                path_to_save = OUTPUT_FOLDER + "gensim/" + get_model_name(k, alpha, beta)

                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

                lda_model.save(path_to_save)

                topics = get_gensim_topics(lda_model, k)

                df = df.append({
                    "k": k,
                    "alpha": alpha,
                    "beta": beta,
                    "model": get_model_name(k, alpha, beta),
                    "gensim_default_coherence": get_coherence_score_gensim(lda_model, documents),
                    "gensim_alt_coherence": get_coherence_score(topics, documents, dictionary, "c_v"),
                    "path": path_to_save
                }, ignore_index=True)

                current_iteration = current_iteration + 1
                update_progress_bar(current_iteration, total_iterations)

    return df


def train_lda(dictionary, documents, topics, alpha_values, beta_values, external_dataset=None, use_cv=True):
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
        "model": [],
        "c_v": [],
        "u_mass": [],
        "c_uci": [],
        "c_npmi": [],
        "diversity": [],
        "path": []
    })

    joined_documents = [ " ".join(document) for document in documents ]

    vectorizer = CountVectorizer(min_df=0.01, max_df=0.85) if (use_cv == True) else CountVectorizer()

    vectorized_documents = vectorizer.fit_transform(joined_documents)

    print(f'Resulting vectorized vocabulary has {len(vectorizer.vocabulary_)} tokens, where {len(vectorizer.stop_words_)} stopwords have been removed')

    vec_docs_path = OUTPUT_FOLDER + "lda_count_vectorized_documents"

    os.makedirs(os.path.dirname(vec_docs_path), exist_ok=True)

    joblib.dump(vectorized_documents, vec_docs_path, compress=7)

    total_iterations = len(topics)

    eval_dataset = documents if (external_dataset == None) else external_dataset

    current_iteration = 0

    update_progress_bar(current_iteration, total_iterations)
    for k in topics:
        lda = LatentDirichletAllocation(
            n_components=k,
            learning_method='online',
            n_jobs=-1,
            random_state=0
        )

        doc_topic_dist = lda.fit_transform(vectorized_documents)
        
        topic_word_dist = lda.components_

        idx_to_word = np.array(vectorizer.get_feature_names())

        topics = get_textual_topics(idx_to_word, topic_word_dist)

        path_to_save = OUTPUT_FOLDER + "scikit/" + get_model_name_for_k(k)

        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

        model = {
            "vectorizer": vectorizer,
            "vectorized_docs_path": vec_docs_path,
            "instance": lda,
            "doc_topic_dist": doc_topic_dist,
            "topic_word_dist": topic_word_dist,
            "idx_to_word": idx_to_word,
            "topics": topics
        }

        joblib.dump(model, path_to_save, compress=7)

        df = df.append({
            "k": k,
            "model": get_model_name_for_k(k),
            "c_v": get_coherence_score(topics, eval_dataset, dictionary, "c_v"),
            "u_mass": get_coherence_score(topics, eval_dataset, dictionary, "u_mass"),
            "c_uci": get_coherence_score(topics, eval_dataset, dictionary, "c_uci"),
            "c_npmi": get_coherence_score(topics, eval_dataset, dictionary, "c_npmi"),
            "diversity": get_topic_diversity(topics),
            "path": path_to_save
        }, ignore_index=True)

        current_iteration = current_iteration + 1
        update_progress_bar(current_iteration, total_iterations)

    return df


def train_many_lda(documents, dictionary, topics, alpha_values, beta_values, corpus=None, external_dataset=None, use_cv=False):
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

    if (corpus != None):
        folder = "gensim/"
        df = train_lda_gensim(corpus, dictionary, documents, topics, alpha_values, beta_values)
    else:
        folder = "scikit/"
        df = train_lda(dictionary, documents, topics, alpha_values, beta_values, external_dataset=external_dataset, use_cv=use_cv)

    training_end_time = time.time()

    total_training_time_in_seconds = training_end_time - training_start_time

    print(f'Total model training time: {str(datetime.timedelta(seconds=total_training_time_in_seconds))}\n')

    print("\nFinishing training...")

    print(df.head())

    print("Saving results CSV...")

    output_filepath = OUTPUT_PATH + f'lda_results.csv'

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    df.to_csv(output_filepath)

    print("CSV file with results saved to ", output_filepath)

    return output_filepath
