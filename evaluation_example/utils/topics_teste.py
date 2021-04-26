import numpy as np
import json
from topics import get_average_topics_vectors, get_most_similar_terms_to_topic
from linguistic_inquiry_word_count import get_categories_for_word

topics = [[(0.0298750012526665, 'mulher'),
  (0.027214789599418917, 'homem'),
  (0.025804695658991867, 'menino'),
  (0.024249028567789814, 'escola'),
  (0.024133272026786615, 'amigo'),
  (0.017434170992274487, 'tempo'),
  (0.016070353498491726, 'amigar'),
  (0.014081989229677407, 'epoca'),
  (0.013752862750655495, 'sala'),
  (0.013364657780353418, 'namorar')],
 [(0.044066698571986196, 'pessoa'),
  (0.041588142344491395, 'coisa'),
  (0.028814016249543096, 'vidar'),
  (0.025979686701184147, 'amigo'),
  (0.02396922583238184, 'tempo'),
  (0.02045630091693466, 'casar'),
  (0.019264476347290563, 'gente'),
  (0.016731557696737728, 'vezar'),
  (0.01253340729034361, 'problema'),
  (0.0121760706169038, 'entao')],
 [(0.051617111858435205, 'vidar'),
  (0.03498769105941979, 'coisa'),
  (0.027319780818512534, 'casar'),
  (0.026933806174256975, 'pessoa'),
  (0.023840409462268786, 'problema'),
  (0.02239847528866488, 'tempo'),
  (0.015789141823188853, 'familia'),
  (0.014892888149390491, 'trabalhar'),
  (0.014501867321418501, 'empregar'),
  (0.013641339870979053, 'dinheiro')],
 [(0.0196553065532863, 'cursar'),
  (0.016427301409889454, 'escola'),
  (0.015869813632703726, 'aula'),
  (0.015312431130093826, 'partir'),
  (0.01403197098157597, 'faculdade'),
  (0.011380990181148498, 'provar'),
  (0.011100769378021746, 'professorar'),
  (0.010368848059451206, 'ensinar'),
  (0.00935922682066401, 'noto'),
  (0.009088264609933262, 'sociedade')],
 [(0.04063401786842112, 'pessoa'),
  (0.03307289189611563, 'suicidio'),
  (0.029965934134374355, 'ansiedade'),
  (0.021619719513263226, 'formar'),
  (0.020853108499889274, 'tratamento'),
  (0.019235841325555208, 'remedios'),
  (0.018812429665021772, 'psiquiatro'),
  (0.015533100421496375, 'pensamento'),
  (0.015036183077073719, 'morte'),
  (0.014560720417824897, 'terapia')]]

word_lemma_file = '/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/datasets/processed/TEST_lemmatized_nouns_only/lemmatized_nouns_only_pt[word_lemma_maps].json'
word_lemma_map = json.load(open(word_lemma_file, 'r'))
original_embeddings_file = '/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/embeddings/ptwiki_20180420_300d.txt'
optimized_embeddings_file = '/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/embeddings/optimized/ptwiki_20180420_300d.w2v'
liwc_path = '/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/liwc/LIWC2007_Portugues_win.dic'

topics_vectors = get_average_topics_vectors(
    topics,
    word_lemma_map['lemma_word'],
    original_embeddings_file,
)

print(f'len(topic_vectors) = {len(topics_vectors)}')
print(f'{topics_vectors}')

for idx, topic in enumerate(topics_vectors):
    most_similar = get_most_similar_terms_to_topic(topic, optimized_embeddings_file, top_n=5)
    print(f'"{most_similar[0]}" is the most similar word to topic "{",".join(list(map(lambda x: x[1], topics[idx])))}"')
    word_categories = get_categories_for_word(most_similar[0][0], liwc_path)
    print(f'liwc categories for "{most_similar[0][0]}": {word_categories}')
