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
    return words_with_values


def get_word_and_vector_from_line(line):
    return line[0], np.array(line[1]).astype(np.float)


def get_topics_vectors(topics, lemma_word_mapping, embeddings_file):
    iterator = MemoryFriendlyFileIterator(embeddings_file)

    topics_vectors = [[] for topic in topics]
    topics_words_missing_from_embeddings = [topic for topic in topics]

    for line in iterator:
        word, vect = get_word_and_vector_from_line(line)
        for i in range(len(topics)):
            if word in topics[i]:
                idx = topics[i].index(word)
                topics_words_missing_from_embeddings[i].remove(word)
                topics_vectors[i].append((idx, vect))
    
    all_topics_words_were_found = all([len(tp) == 0 for tp in topics_words_missing_from_embeddings])
    if all_topics_words_were_found:
        return topics_vectors
    
    iterator = MemoryFriendlyFileIterator(embeddings_file)

    for line in iterator:
        word, vect = get_word_and_vector_from_line(line)
        for i in range(len(topics_words_missing_from_embeddings)):
            for lemma in topics_words_missing_from_embeddings[i]:
                if word in lemma_word_mapping[lemma]:
                    idx = topics[i].index(lemma)
                    topics_vectors[i].append((idx, vect))

    flattened_topics_vectors = []
    for topic in topics_vectors:
        topic.sort(key=idx_sort)
        flattened_topics_vectors.append(list(map(lambda x: x[1], topic)))
    
    return flattened_topics_vectors


def get_average_topics_vectors(topics, lemma_word_mapping, embeddings_file):
    topics_vectors = get_topics_vectors(topics, lemma_word_mapping, embeddings_file)

    return [np.mean(topic_vectors, axis=0) for topic_vectors in topics_vectors]


def similarity_sort(element):
    return element[1]


def idx_sort(element):
    return element[0]


def get_most_similar_terms_to_topic(topic_vector, embeddings_file, top_n = 10):
    iterator = MemoryFriendlyFileIterator(embeddings_file)

    most_similar_terms = []

    for line in iterator:
        word, word_vector = get_word_and_vector_from_line(line)

        similarity = np.average(cosine_similarity(topic_vector, word_vector))

        if len(most_similar_terms) == 0:
            most_similar_terms.append((word, similarity))
        elif similarity > most_similar_terms[0][1]:
            most_similar_terms.pop(0)
            most_similar_terms.append((word, similarity))
            most_similar_terms.sort(key=similarity_sort)
        
        most_similar_terms = most_similar_terms[-top_n:]
        
    return most_similar_terms
