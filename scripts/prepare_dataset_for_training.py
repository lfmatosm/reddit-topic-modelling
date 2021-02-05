from sklearn.feature_extraction.text import CountVectorizer
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from gensim.corpora import Dictionary
import json
import numpy as np
from scipy import sparse
import os
import argparse
import joblib


def remove_empty(in_docs):
    return [doc for doc in in_docs if doc!=[]]


def _to_numpy_array(documents):
    return np.array([[np.array(doc) for doc in documents]],
                    dtype=object).squeeze()

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


parser = argparse.ArgumentParser(description='Splits dataset for LDA, ETM and CTM models training.')

parser.add_argument('--dataset', type=str, help='dataset path', required=True)
parser.add_argument('--dataset_name', type=str, help='dataset path', default='training_data', required=False)
parser.add_argument('--min_df', default=0.01, type=float, help='minimum document frequency for document vectorizer', required=False)
parser.add_argument('--max_df', default=0.85, type=float, help='minimum document frequency for document vectorizer', required=False)

args = parser.parse_args()

min_df=args.min_df
max_df=args.max_df
dataset=args.dataset
dataset_name = args.dataset_name

ADDITIONAL_STOPWORDS = ["http", "https", "watch", "comment", "comments"]

print(f'dataset: {dataset}\ndataset_name: {dataset_name}\nmin_df: {min_df}\nmax_df: {max_df}\n')

OUTPUT_PATH = f'./resources/{dataset_name}'

print("Loading dataset file...")
documents = json.load(open(dataset, "r"))
joined_documents = [ " ".join(document["body"]) for document in documents ]
print(f'Dataset length: {len(joined_documents)}')

print("Counting word frequencies...")
vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
print("Word frequencies computed")

print("Vectorizing documents...")
vectorized_documents = vectorizer.fit_transform(joined_documents)
print("Vecorized documents")

print(f'Vectorizer vocab length: {len(vectorizer.vocabulary_)}')
print(f'Vectorizer stopwords length: {len(vectorizer.stop_words_)}')

vectorizer_path = OUTPUT_PATH + "/cvectorizer.cvec"
os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
joblib.dump(vectorizer, vectorizer_path, compress=7)

print("Filtering stopwords from documents...")
documents_without_most_frequent = [
    [word for word in document.split() \
        if word not in vectorizer.stop_words_ and word not in ADDITIONAL_STOPWORDS] \
            for document in joined_documents]
print("Stopwords filtered from documents...")

print(f'Documents length after filtering: {len(documents_without_most_frequent)}')

docs = documents_without_most_frequent

cvz = vectorized_documents.sign()

# Get vocabulary
print('Building vocabulary...')
sum_counts = cvz.sum(axis=0)
v_size = sum_counts.shape[1]
sum_counts_np = np.zeros(v_size, dtype=int)
for v in range(v_size):
    sum_counts_np[v] = sum_counts[0,v]
word2id = dict([(w, vectorizer.vocabulary_.get(w)) for w in vectorizer.vocabulary_])
id2word = dict([(vectorizer.vocabulary_.get(w), w) for w in vectorizer.vocabulary_])
del vectorizer
print('Initial vocabulary size: {}'.format(v_size))

# Sort elements in vocabulary
idx_sort = np.argsort(sum_counts_np)
vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]

# Create dictionary and inverse dictionary
vocab = vocab_aux
del vocab_aux
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])

print('Tokenizing documents and creating train dataset...')
num_docs = cvz.shape[0]
trSize = num_docs
del cvz
idx_permute = np.random.permutation(num_docs).astype(int)

# Remove words not in train_data
vocab = list(set([w for idx_d in range(trSize) for w in docs[idx_permute[idx_d]] if w in word2id]))
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])
print('vocabulary after removing words not in train: {}'.format(len(vocab)))

docs_tr = [[word2id[w] for w in docs[idx_permute[idx_d]] if w in word2id] for idx_d in range(trSize)]
del docs

print('Number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))

# Remove empty documents
print('Removing empty documents...')

docs_tr = remove_empty(docs_tr)

# Obtains the training, test and validation datasets as word lists
words_tr = [[id2word[w] for w in doc] for doc in docs_tr]

joined_words_tr = list(map(lambda x: " ".join(x), words_tr))
documents_path = OUTPUT_PATH + "/joined_documents.json"
print(f'Saving joined documents file with {len(joined_words_tr)} documents (JSON): {documents_path}')
os.makedirs(os.path.dirname(documents_path), exist_ok=True)
json.dump(joined_words_tr, open(documents_path, 'w'))
print(f'Joined documents file saved (JSON): {documents_path}')

path_save = OUTPUT_PATH + '/split_documents.json' #training_dataset.json
print(f'Saving split documents (JSON) with {len(words_tr)} documents: {path_save}')
os.makedirs(os.path.dirname(path_save), exist_ok=True)
json.dump(words_tr, open(path_save, 'w'))
print(f'Split documents saved (JSON): {path_save}')


print("Creating word dictionary for entire corpus...")
dictionary = create_dictionary(words_tr)
path_save = OUTPUT_PATH + '/word_dictionary.gdict'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(dictionary, path_save, compress=7)
print(f'Word dictionary created and saved to "{path_save}"')

# Getting lists of words and doc_indices
print('(ETM) Creating lists of words...')

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]

words_tr = create_list_words(docs_tr)

print('  len(words_tr): ', len(words_tr))

# Get doc indices
print('(ETM) Getting doc indices...')

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

doc_indices_tr = create_doc_indices(docs_tr)

print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))

# Number of documents in each set
n_docs_tr = len(docs_tr)

# Remove unused variables
del docs_tr

# Create bow representation
print('(ETM) Creating bow representation...')

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))

del words_tr
del doc_indices_tr

# Split bow intro token/value pairs
print('(ETM) Splitting bow intro token/value pairs and saving...')

def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    return indices, counts

bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)

etm_training_dataset = {
    "tokens": _to_numpy_array(bow_tr_tokens), 
    "counts": _to_numpy_array(bow_tr_counts),
}

print("Creating ETM vocabulary file...")
path_save = OUTPUT_PATH + '/etm_vocabulary.vocab'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(vocab, path_save, compress=7)
print(f'ETM vocabulary saved to "{path_save}"')

print("Creating ETM training dataset file...")
path_save = OUTPUT_PATH + '/etm_training_dataset.dataset'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(etm_training_dataset, path_save, compress=7)
print(f'ETM training dataset saved to "{path_save}"')


print("Creating CTM training dataset file...")
simple_preprocessing = WhiteSpacePreprocessing(joined_words_tr, "portuguese")
preprocessed_documents_for_bow, unpreprocessed_corpus_for_contextual, vocab = simple_preprocessing.preprocess()

data_preparation = TopicModelDataPreparation("distiluse-base-multilingual-cased")
ctm_training_dataset = data_preparation.create_training_set(unpreprocessed_corpus_for_contextual, preprocessed_documents_for_bow)

path_save = OUTPUT_PATH + '/ctm_data_preparation.obj'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(data_preparation, path_save, compress=7)
print(f'CTM data preparation instance saved to "{path_save}"')

path_save = OUTPUT_PATH + '/ctm_training_dataset.dataset'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(ctm_training_dataset, path_save, compress=7)
print(f'CTM training dataset saved to "{path_save}"')


print('\nDatasets prepared for training')
