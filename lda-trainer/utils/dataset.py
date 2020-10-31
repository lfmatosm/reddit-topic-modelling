from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import json


def load_dataset(dataset_path, field_to_read=None):
    dataset = json.load(open(dataset_path, 'r'))

    if field_to_read != None:
        return [data[field_to_read].split() \
            if (isinstance(data[field_to_read], str)) \
            else data[field_to_read] for data in dataset]

    return [data for data in dataset]


def create_bigram(documents):
    """Creates bigram pairs for a dataset of documents.

    Parameters:
    
    documents (list of str): set of documents

    Returns:

    list of str: bigram pairs
    """
    bigram = Phrases(documents, min_count=3, threshold=5)
    
    bigram_mod = Phraser(bigram)

    return [bigram_mod[document] for document in documents]


def create_dictionary(documents):
    """Creates word dictionary for given corpus.

    Parameters:
    
    documents (list of str): set of documents

    Returns:

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset
    """
    dictionary = Dictionary(documents)
    dictionary.compactify()

    # Uncomment the line below if you want to keep a proportion of the tokens in the dictionary
    # dictionary.filter_extremes(no_below=1, no_above=1.0)

    return dictionary


def create_corpus(documents, dictionary):
    """Creates BOW (bag-of-words) corpus for a set of documents and its word dictionary.

    Parameters:
    
    documents (list of str): set of documents

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset

    Returns:

    list of int list: each document codified as BOW, e.g. each text word is associated with a number across corpus
    """
    return [dictionary.doc2bow(text) for text in documents]


def load_textual_dataset(dataset_file):
    with open(dataset_file, 'r') as data:
        for line in data:
            yield line.split()
