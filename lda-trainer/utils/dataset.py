from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models.phrases import Phraser


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
