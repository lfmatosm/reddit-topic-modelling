from scipy.spatial import distance
from itertools import combinations
import numpy as np

topics = [
  ["depressao", "saude", "terapia", "suicidio", "confusao", "problema"],
  ["juizo", "paz", "amor", "sandice", "religiao", "salvacao"],
  ["esporte", "superacao", "depressao", "saude", "amor", "mouse"]
]

wv = {
  "depressao": [0.1, 0.2, 0.3],
  "saude": [0.1, 0.15, 0.34],
  "terapia": [0.6, 0.001, 0.5],
  "suicidio": [0.17, 0.1, 0.13],
  "confusao": [0.1, 0.2, 0.3],
  "problema": [0.234, 0.2, 0.2453],
  "juizo": [0.456, 0.1231, 0.3],
  "paz": [0.3845936, 0.2, 0.3],
  "amor": [0.386456, 0.2, 0.3],
  "sandice": [0.90697, 0.2, 0.3],
  "religiao": [0.1, 0.2, 0.367647],
  "salvacao": [0.1, 0.2342, 0.3],
  "mouse": [0.1, 0.245, 0.3],
  "superacao": [0.786, 0.2678, 0.1345],
  "esporte": [0.4, 0.12, 0.578]
}
# wv = {
#   "depressao": [0.1, 0.2, 0.3],
#   "saude": [0.1, 0.15, 0.34],
#   "terapia": [0.6, 0.001, 0.5],
#   "suicidio": [0.17, 0.1, 0.13],
#   "confusao": [0.1, 0.2, 0.3],
#   "problema": [0.1, 0.2, 0.3],
#   "juizo": [0.1, 0.2, 0.3],
#   "paz": [0.1, 0.2, 0.3],
#   "amor": [0.1, 0.2, 0.3],
#   "sandice": [0.1, 0.2, 0.3],
#   "religiao": [0.1, 0.2, 0.3],
#   "salvacao": [0.1, 0.2, 0.3],
#   "mouse": [0.1, 0.2, 0.3],
#   "superacao": [0.1, 0.2, 0.3],
#   "esporte": [0.1, 0.2, 0.3]
# }
# wv = {
#   "depressao": [0.1, 0.2, 0.3],
#   "saude": [0.1, 0.2, 0.3],
#   "terapia": [0.1, 0.2, 0.3],
#   "suicidio": [0.1, 0.2, 0.3],
#   "confusao": [0.1, 0.2, 0.3],
#   "problema": [0.1, 0.2, 0.3],
#   "juizo": [0.1, 0.2, 0.3],
#   "paz": [0.1, 0.2, 0.3],
#   "amor": [0.1, 0.2, 0.3],
#   "sandice": [0.1, 0.9, 0.3],
#   "religiao": [0.1, 0.2, 0.3],
#   "salvacao": [0.1, 0.2, 0.3],
#   "mouse": [0.1, 0.2, 0.3],
#   "superacao": [0.1, 0.2, 0.3],
#   "esporte": [0.1, 0.2, 0.3]
# }

def pairwise_word_embedding_distance(topics, wv, topk=10):
    """
    :param topk: how many most likely words to consider in the evaluation
    :return: topic coherence computed on the word embeddings similarities
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        count = 0
        sum_dist = 0
        for list1, list2 in combinations(topics, 2):
            count = count+1
            word_counts = 0
            dist = 0
            for word1 in list1[:topk]:
                for word2 in list2[:topk]:
                    dist = dist + distance.cosine(wv[word1], wv[word2])
                    word_counts = word_counts + 1

            dist = dist/word_counts
            sum_dist = sum_dist + dist
        return sum_dist/count

print(pairwise_word_embedding_distance(topics, wv, topk=6))

def pairwise_word_embedding_distance2(topics, wv, topk=10):
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
            dist += distance.cosine(wv[word1], wv[word2])
          sum_dist += dist / topk
        return sum_dist / len(topics)

print(pairwise_word_embedding_distance2(topics, wv, topk=6))


def normalize(word_embedding):
  return word_embedding / np.sum(word_embedding)


def teste1(topics, wv, topk=10):
    """
    :param topk: how many most likely words to consider in the evaluation
    :return: topic coherence computed on the word embeddings similarities
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        dist = 0
        for topic in topics:
          combs = combinations(topic, 2)
          for word1, word2 in combs:
            w1 = normalize(np.array(wv[word1]))
            w2 = normalize(np.array(wv[word2]))
            dist = dist + np.inner(w1, w2)

        return dist / (topk*(topk-1))

print(teste1(topics, wv, topk=6))