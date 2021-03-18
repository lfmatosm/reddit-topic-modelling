import re
import os
from gensim.models import KeyedVectors

class MemoryFriendlyFileIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


temp_file = "/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/embeddings/temp.txt"
output_file = "/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/embeddings/ptwiki_20180420_300d_optimized.txt"
emb_file = "/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/embeddings/ptwiki_20180420_300d.txt"


entity_count = 0
lines_count = 0
word_count = 0

pattern = re.compile("ENTITY")

iterator = MemoryFriendlyFileIterator(emb_file)
with open(temp_file, 'w') as f:
    for line in iterator:
        lines_count += 1
        if pattern.search(line[0]) is not None or len(line[1:]) != 300:
            entity_count += 1
        else:
            word_count += 1
            f.write(f'{" ".join(line)}\n')

print(f'\n\n{entity_count}/{lines_count}')
print(f'word_count: {word_count}')

del iterator

iterator = MemoryFriendlyFileIterator(temp_file)
with open(output_file, 'w') as f:
    f.write(f'{word_count} 300\n')
    for line in iterator:
        f.write(f'{" ".join(line)}\n')

del iterator

print("Creating gensim embeddings")
embeddings = KeyedVectors.load_word2vec_format(
    output_file, 
    binary=False
)
print(f'Loaded "{output_file}" embeddings')
path = output_file.replace('.txt', '.w2v')
os.makedirs(os.path.dirname(path), exist_ok=True)
embeddings.save(path)
del embeddings
print("Gensim embeddings created")
