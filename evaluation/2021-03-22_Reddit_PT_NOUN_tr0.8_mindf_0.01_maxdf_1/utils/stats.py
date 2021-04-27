from decimal import Decimal
import functools


def get_word_probabilities(idx2word, topic_word_dist, topic_word_mtx):
    word_probabilities = {}

    for i in range(len(topic_word_dist)):
        for j in range(len(topic_word_dist[i])):
            word = topic_word_mtx[i][j]
            if word in word_probabilities:
                word_probabilities[word] = word_probabilities[word] + topic_word_dist[i][j]
            else:
                word_probabilities[word] = topic_word_dist[i][j]
    
    # Sorts the dictionary by values before returning it
    return {k: v for k, v in sorted(word_probabilities.items(), key=lambda item: item[1], reverse=True)}


def merge_word_probabilities(*args):
    return functools.reduce(
        lambda dict1, dict2 : {x: Decimal(dict1.get(x, 0)) + Decimal(dict2.get(x, 0)) \
        for x in set(dict1).union(dict2)}, args
    )