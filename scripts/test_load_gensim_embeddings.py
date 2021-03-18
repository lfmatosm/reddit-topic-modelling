import numpy as np
from gensim.models import KeyedVectors

topic = np.array([0.5] * 300)
embeddings_file = "/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/embeddings/ptwiki_20180420_300d_optimized.w2v"
# embeddings_file = "/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/embeddings/optimized/ptwiki_20180420_300d.w2v"


embeddings = KeyedVectors.load(embeddings_file, mmap='r')
print(f'most similar to "{topic}": {embeddings.most_similar(positive=[topic], topn=20)}')
