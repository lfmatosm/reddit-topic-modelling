from gensim.models import CoherenceModel
import pandas as pd, functools
from collections import Counter
from decimal import Decimal


def get_coherence_score_gensim(model, documents):
    coherence_model = CoherenceModel(
                model=model,
                texts=documents,
                coherence='c_v'
    )

    return coherence_model.get_coherence()


def get_coherence_score(topics, documents, dictionary, coherence):
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
                coherence=coherence
    )

    return coherence_model.get_coherence()


def get_topic_diversity(topics):
    """Calculates topic diversity for given topics. Topic diversity is defined as 
    the frequency in which words are associated with a single topic

    Parameters:

    topics (list of str list): topic words for each topic

    Returns:

    float: diversity score
    """
    word_frequencies_per_topic = [Counter(topic[:20]) for topic in topics]
    word_frequencies = functools.reduce(lambda dict1, dict2 : {x: Decimal(dict1.get(x, 0)) + Decimal(dict2.get(x, 0)) \
        for x in set(dict1).union(dict2)}, word_frequencies_per_topic)
    
    topic_diversity_scores = []
    for topic in topics:
        topic_diversity = 0
        for i in range(0, 20):
            if word_frequencies[topic[i]] == Decimal(1):
                topic_diversity = topic_diversity + 1
        
        topic_diversity_scores.append(Decimal(topic_diversity) / Decimal(20))
    
    return float(Decimal(sum(topic_diversity_scores)) / Decimal(len(topic_diversity_scores)))
