from gensim.models import CoherenceModel
import functools
from collections import Counter
from decimal import Decimal
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
from itertools import combinations
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.word_embeddings_300d
collection = db.embeddings


def get_coherence_score(topics, documents, dictionary, coherence, no_of_words=20):
    """Calculates topic coherence using gensim's coherence pipeline.

    Parameters:

    topics (list of str list): topic words for each topic
    
    documents (list of str): set of documents

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset

    coherence (str): coherence type. Can be 'c_v', 'u_mass', 'c_uci' or 'c_npmi'

    Returns:

    float: coherence score
    """
    coherence_model = CoherenceModel(
                topics=topics, 
                texts=documents, 
                dictionary=dictionary, 
                coherence=coherence,
                processes=0,
                topn=no_of_words
    )

    return coherence_model.get_coherence()


def get_topic_diversity(topics, no_of_words=20):
    """Calculates topic diversity for given topics. Topic diversity is defined as 
    the frequency in which words are associated with a single topic

    Parameters:

    topics (list of str list): topic words for each topic

    Returns:

    float: diversity score
    """
    word_frequencies_per_topic = [Counter(topic[:no_of_words]) for topic in topics]
    word_frequencies = functools.reduce(lambda dict1, dict2 : {x: Decimal(dict1.get(x, 0)) + Decimal(dict2.get(x, 0)) \
        for x in set(dict1).union(dict2)}, word_frequencies_per_topic)
    
    topic_diversity_scores = []
    for topic in topics:
        topic_diversity = 0
        for i in range(0, no_of_words):
            if word_frequencies[topic[i]] == Decimal(1):
                topic_diversity = topic_diversity + 1
        
        topic_diversity_scores.append(Decimal(topic_diversity) / Decimal(no_of_words))
    
    return float(Decimal(sum(topic_diversity_scores)) / Decimal(len(topic_diversity_scores)))


def create_dictionary(documents):
    """Creates word dictionary for given corpus.

    Parameters:
    
    documents (list of str): set of documents

    Returns:

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset
    """
    dictionary = Dictionary(documents)
    dictionary.compactify()

    return dictionary


def filter_tokens_by_frequencies(documents, min_df=1, max_df=1.0):
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    vectorizer.fit_transform(documents)
    
    return [[word for word in document if word not in vectorizer.stop_words_] for document in documents]


def get_words_from_db(word1, word2):
    print(f'word1={word1}, word2={word2}')
    w1 = collection.find_one({ "word": word1 })
    w2 = collection.find_one({ "word": word2 })

    print(f'w1={w1}, w1={w2}')

    def f(x): return float(x)

    return list(map(f, w1)), list(map(f, w2))


def pairwise_word_embedding_distance(topics, topk=20):
    """
    :param topk: how many most likely words to consider in the evaluation
    :return: topic coherence computed on the word embeddings similarities
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        sum_dist = 0
        for topic in topics:
            dist = 0
            combs = combinations(topic[:topk], 2)
            for word1, word2 in combs:
                w1, w2 = get_words_from_db(word1, word2)
                dist += distance.cosine(w1, w2)
            sum_dist += dist / topk
        return sum_dist / len(topics)


def get_topic_word_matrix(topic_word_mtx, k, idx2token):
    topics = []
    for i in range(k):
        words_dists = list(topic_word_mtx[i].cpu().numpy())
        component_words = [idx2token[idx]
                            for idx, _ in enumerate(words_dists)]
        topics.append(component_words)
    return topics
