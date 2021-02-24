import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
    return list(map(lambda x: x[0], words_with_values))


def get_topics_vectors(topics, lemma_word_mapping, embeddings_file):
    iterator = MemoryFriendlyFileIterator(embeddings_file)

    topics_vectors = [[] for topic in topics]

    words_with_lemmatized_values = [get_words_with_lemmatized_values_in_topic(lemma_word_mapping, topic) \
        for topic in topics]

    for line in iterator:
        word = line[0]
        vect = np.array(line[1:]).astype(np.float)
        for i in range(len(topics)):
            if word in topics[i] or word in words_with_lemmatized_values[i]:
                topics_vectors[i].append(vect)
        
    return topics_vectors


def get_average_topics_vectors(topics, lemma_word_mapping, embeddings_file):
    topics_vectors = get_topics_vectors(topics, lemma_word_mapping, embeddings_file)

    return [np.mean(topic_vectors, axis=0) for topic_vectors in topics_vectors]


def similarity_sort(element):
    return element[1]


def get_most_similar_terms_to_topic(topic_vector, embeddings_file, top_n = 10):
    iterator = MemoryFriendlyFileIterator(embeddings_file)

    most_similar_terms = []

    for line in iterator:
        word = line[0]
        word_vector = np.array(line[1:]).astype(np.float)

        similarity = np.average(cosine_similarity(topic_vector, word_vector))

        if len(most_similar_terms) == 0:
            most_similar_terms.append((word, similarity))
        elif similarity > most_similar_terms[0][1]:
            most_similar_terms.pop(0)
            most_similar_terms.append((word, similarity))
            most_similar_terms.sort(key=similarity_sort)
        
        most_similar_terms = most_similar_terms[-top_n:]
        
    return most_similar_terms
