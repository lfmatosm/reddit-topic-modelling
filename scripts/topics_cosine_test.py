from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

X = [1, 1, 1]
Y = [1, 1, 1]

print(f'cos = {cosine_similarity(X, Y)}\n')
