import numpy as np
import copy
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors


class MemoryFriendlyFileIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


def get_words_with_lemmatized_values_in_topic(lemma_word_map, topic):
    def word_has_value_in_topic(mapping):
        _, words = mapping
        for word in words:
            if word in topic:
                return True
        return False
    
    words_with_values = list(filter(word_has_value_in_topic, list(lemma_word_map.items())))
    return words_with_values


def get_word_and_vector_from_line(line):
    return line[0], np.array(line[1:]).astype(np.float)


def get_topics_vectors(topics, lemma_word_mapping, embeddings_file):
    iterator = MemoryFriendlyFileIterator(embeddings_file)

    topics_words = [list(map(lambda x: x[1], topic)) for topic in topics]

    word_vectors = [[] for topic in topics]
    word_weights = [[] for topic in topics]
    
    topics_words_missing_from_embeddings = copy.deepcopy(topics)

    for line in iterator:
        for i in range(len(topics_words)):
            word = line[0]
            if line[0] in topics_words[i]:
                vect = np.array(line[1:]).astype(np.float)
                idx = topics_words[i].index(word)
                word_prob_pair = topics[i][idx]
                topics_words_missing_from_embeddings[i].remove(word_prob_pair)
                (prob, _) = word_prob_pair
                word_vectors[i].append((idx, vect))
                word_weights[i].append((idx, prob))
    
    all_topics_words_were_found = all([len(tp) == 0 for tp in topics_words_missing_from_embeddings])
    if all_topics_words_were_found:
        return word_vectors
    
    missing_words = [list(map(lambda x: x[1], topic)) for topic in topics_words_missing_from_embeddings]
    iterator = MemoryFriendlyFileIterator(embeddings_file)

    lemmas_already_added = [[] for topic in topics]

    for line in iterator:
        for i in range(len(missing_words)):
            for lemma in missing_words[i]:
                word = line[0]
                if lemma not in lemmas_already_added[i] and word in lemma_word_mapping[lemma]:
                    vect = np.array(line[1:]).astype(np.float)
                    idx = topics_words[i].index(lemma)
                    word_prob_pair = topics[i][idx]
                    (prob, _) = word_prob_pair
                    word_vectors[i].append((idx, vect))
                    word_weights[i].append((idx, prob))
                    lemmas_already_added[i].append(lemma)

    flattened_word_vectors = []
    for topic in word_vectors:
        topic.sort(key=idx_sort)
        flattened_word_vectors.append(list(map(lambda x: x[1], topic)))
    
    flattened_word_weights = []
    for topic in word_weights:
        topic.sort(key=idx_sort)
        flattened_word_weights.append(list(map(lambda x: x[1], topic)))
    
    return flattened_word_vectors, flattened_word_weights


def get_topics_vectors_gensim(topics, lemma_word_mapping, embeddings_file):
    embeddings = KeyedVectors.load(embeddings_file, mmap='r')

    topics_words = [list(map(lambda x: x[1], topic)) for topic in topics]

    word_vectors = [[] for topic in topics]
    word_weights = [[] for topic in topics]
    
    topics_words_missing_from_embeddings = copy.deepcopy(topics)

    for idx, topic in enumerate(topics_words):
        for i in range(len(topic)):
            word = topic[i]
            try:
                vector = embeddings[word]
                word_prob_pair = topics[idx][i]
                (prob, _) = word_prob_pair
                word_vectors[idx].append((i, vector))
                word_weights[idx].append((i, prob))
            except:
              for original_word in lemma_word_mapping[word]:
                try:
                    vector = embeddings[word]
                    word_prob_pair = topics[idx][i]
                    (prob, _) = word_prob_pair
                    word_vectors[idx].append((i, vector))
                    word_weights[idx].append((i, prob))
                except: 
                    continue

    flattened_word_vectors = []
    for topic in word_vectors:
        topic.sort(key=idx_sort)
        flattened_word_vectors.append(list(map(lambda x: x[1], topic)))
    
    flattened_word_weights = []
    for topic in word_weights:
        topic.sort(key=idx_sort)
        flattened_word_weights.append(list(map(lambda x: x[1], topic)))
    
    return flattened_word_vectors, flattened_word_weights


def get_average_topics_vectors(topics, lemma_word_mapping, embeddings_file):
    word_vectors, word_weights = get_topics_vectors(topics, lemma_word_mapping, embeddings_file)

    weighted_topics = []
    for idx, topic in enumerate(word_vectors):
        word_weight = word_weights[idx]
        weighted_topic = [np.multiply(topic[i], word_weight[i]) for i in range(len(topic))]
        weighted_topics.append(weighted_topic)

    return [np.mean(word_vectors, axis=0) for word_vectors in weighted_topics]


def similarity_sort(element):
    return element[1]


def idx_sort(element):
    return element[0]


def get_most_similar_terms_to_topic_gensim(topic_vector, embeddings_file, top_n = 10):
    embeddings = KeyedVectors.load(embeddings_file, mmap='r')
    most_similar_words = embeddings.most_similar(positive=[topic_vector], topn=20)
    return list(filter(lambda x: not x[0].lower().startswith('entity'), most_similar_words))[:top_n]


def get_most_similar_terms_to_topic(topic_vector, embeddings_file, top_n = 10, use_gensim = True):
    if use_gensim:
        return get_most_similar_terms_to_topic_gensim(topic_vector, embeddings_file, top_n)
    
    iterator = MemoryFriendlyFileIterator(embeddings_file)

    most_similar_terms = []

    for line in iterator:
        if line[0].lower().startswith('entity'): continue
        word, word_vector = get_word_and_vector_from_line(line)
        
        similarity = np.subtract(1, cosine(topic_vector, word_vector))

        if len(most_similar_terms) == 0:
            most_similar_terms.append((word, similarity))
        else:
            for i in range(len(most_similar_terms), -1, -1):
                if similarity > most_similar_terms[i][1]:
                    most_similar_terms[i] = (word, similarity)
                    break
        
        most_similar_terms = most_similar_terms[-top_n:]
        
    return most_similar_terms[::-1]


def get_word_probability_mappings(topics):
    mappings = [dict(topic) for topic in topics]
    return [{v: k for k, v in mapping.items()} for mapping in mappings]
