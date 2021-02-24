from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = {}
with open("scripts/etm_w2v_embedding.txt", "r") as file:
    for line in file.readlines():
        splitted = line.split()
        word = splitted[0]
        embeddings[word] = np.array([float(n) for n in splitted[1:]])

# print(f'Embeddings: {list(embeddings.keys())[:5]}')

topics = []
with open("scripts/topics.txt", "r") as file:
    for line in file.readlines():
        topics.append(line.split())
print(f'Topics: {topics[:5]}')

topic_embeddings = [[embeddings[word] for word in topic] for topic in topics]
# print(f'Topic embeddings: {topic_embeddings[:5]}')
print(f'Topic embeddings length: {len(topic_embeddings)}')

combs = list(combinations(range(len(topic_embeddings)), 2))
# print(f'total combinations = {list(combs)}')

print(f'combs length = {len(combs)}')
similarities = np.array([])
for xi, yi in combs:
    print(f'xi={xi}')
    print(f'yi={yi}')
    similarity = np.average(cosine_similarity(topic_embeddings[xi], topic_embeddings[yi]))
    print(f'avg similarity = {similarity}')
    print(f'avg cos = {np.average(similarity)}')
    similarities = np.append(similarities, similarity)
print(f'similarities length = {len(similarities)}')
print(similarities)
max_idx = np.argmax(similarities)
print(f'max idx similarities = {max_idx}')
print(f'max similarity = {similarities[max_idx]}')
first_topic_idx, second_topic_idx = combs[max_idx]
print(f'best comb = first: {first_topic_idx}, second: {second_topic_idx}')
print(f'most similar topics: 1 - {", ".join(topics[first_topic_idx])}\n2 - {", ".join(topics[second_topic_idx])}\n')

# X = np.array([[1, 1, 1], [0.98, 0.1, 0.21], [0, 0, 0], [0.8, 0, 1]])
# Y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0.99, 1, 0.7]])

# print(X[0:1])

# similarity = cosine_similarity(X, Y)
# print(f'avg cos = {np.average(similarity)}')
# print(f'cos = {similarity}\n')
