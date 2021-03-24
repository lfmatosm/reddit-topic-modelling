from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
import json
import numpy as np
from scipy import sparse
import os
import argparse
import joblib
import time
import logging


LOG_FOLDER = "pipeline_logs/"

class MemoryFriendlyFileIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


def remove_empty(in_docs):
    return [doc for doc in in_docs if len(doc) > 0]


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
    # dictionary.compactify()

    # Uncomment the line below if you want to keep a proportion of the tokens in the dictionary
    dictionary.filter_extremes(no_below=1, no_above=1.0)

    return dictionary


def get_original_keys_with_lemmatized_values_in_vocabulary(lemma_word_map, vocabulary):
    def key_has_value_in_vocab(mapping):
        _, words = mapping
        for word in words:
            if word in vocabulary:
                return True
        return False
    
    keys_with_values = list(filter(key_has_value_in_vocab, list(lemma_word_map.items())))
    return list(map(lambda x: x[0], keys_with_values))


def optimize_embeddings(
    vocabulary, 
    lemma_word_mapping,
    embedding_file,
    output_embedding_path, 
    n_dim,
    logging,
):
    original_embeddings = KeyedVectors.load(embedding_file, mmap='r')
    embeddings_redux = KeyedVectors(n_dim)

    words = []
    weights = []

    logging.info("Generating optimized W2V embedding based on vocabulary words...")

    count = 0
    for word in vocabulary:
        try:
            vector = original_embeddings[word]
            words.append(word)
            weights.append(vector)
            count += 1
        except:
            logging.info(f'Embeddings: word "{word}" not found on embeddings!')
            pass

    del original_embeddings
    embeddings_redux.add(words, weights)
    del words
    del weights

    os.makedirs(os.path.dirname(output_embedding_path), exist_ok=True)
    embeddings_redux.save(output_embedding_path)
    del embeddings_redux

    logging.info(f'\n\nGenerated optimized Gensim W2V embedding file at "{output_embedding_path}"')
    del output_embedding_path
    logging.info(f'{count}/{len(vocabulary)} words found on embeddings')


parser = argparse.ArgumentParser(description='Prepares training/testing resources for LDA/CTM/ETM training scripts')
parser.add_argument('--dataset', type=str, help='dataset path', required=True)
parser.add_argument('--word_lemma_maps', type=str, help='word-lemma mappings path', required=False, default=None)
parser.add_argument('--dictionary', type=str, help='dictionary path', required=True)
parser.add_argument('--embeddings', type=str, help='embeddings path', required=True)
parser.add_argument('--n_dim', type=int, help='embeddings dimensions', required=False, default=300)
parser.add_argument('--train_size', type=float, help='train size', required=False, default=1.0)
parser.add_argument('--dataset_name', type=str, help='dataset name', default='training_data', required=False)
parser.add_argument('--lang', type=str, help='dataset language', required=True, default='en')
args = parser.parse_args()

LOG_FILE = os.path.join(LOG_FOLDER, f'{args.lang}_preparation_{args.dataset_name}.txt')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

start = time.time()

dataset=args.dataset
dataset_name = args.dataset_name
train_size = float(args.train_size)

logging.info(f'Dataset: {dataset}\nDataset_name: {dataset_name}\nTrain_size: {train_size}\n\n')

OUTPUT_PATH = f'./resources/{dataset_name}'

logging.info("Loading dataset file...")
documents = json.load(open(dataset, "r"))
joined_documents = [ " ".join(document["body"]) for document in documents ]
logging.info(f'Dataset length: {len(joined_documents)}')
del documents

logging.info("Loading Gensim dictionary...")
dictionary = joblib.load(args.dictionary)
logging.info(f'No. of tokens on dictionary: {len(dictionary.token2id)}')
vocab = [key for key in dictionary.token2id.keys()]
logging.info(f'Gensim vocabulary length: {len(vocab)}')
word2id = dictionary.token2id
id2word = {v: k for k, v in word2id.items()}

word_lemma_maps = None
if args.word_lemma_maps is not None:
    logging.info("Loading word-lemma mappings...")
    word_lemma_maps = json.load(open(args.word_lemma_maps, "r"))
    logging.info(f'Word-lemma and inverse mappings loaded with {len(word_lemma_maps)} entries')


logging.info("Filtering words not present in vocabulary...")
documents_without_stopwords = [
    [word for word in document.split() \
        if word in vocab] \
            for document in joined_documents]
documents_without_stopwords = [document for document in documents_without_stopwords if len(document) > 0]
logging.info("Filtered non-vocabulary words")

logging.info(f'Documents length after filtering and removal of empty documents: {len(documents_without_stopwords)}')

docs = documents_without_stopwords

logging.info('Tokenizing documents and creating train dataset...')
num_docs = len(docs)
trSize = int(np.floor(train_size*num_docs))
tsSize = int(num_docs - trSize)
logging.info(f'No. documents - Train: {trSize}\tTest: {tsSize}')

idx_permute = np.random.permutation(num_docs).astype(int)

docs_tr = [
    [word2id[w] for w in docs[idx_permute[idx_d]] if w in word2id] \
        for idx_d in range(trSize)
]
docs_ts = [
    [word2id[w] for w in docs[idx_permute[idx_d]] if w in word2id] \
        for idx_d in range(tsSize)
]
del word2id
del idx_permute
del docs

logging.info('Number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
logging.info('Number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
del tsSize
del trSize

# Obtains the training snd test datasets as word lists
words_tr = [[id2word[w] for w in doc] for doc in docs_tr]
words_ts = [[id2word[w] for w in doc] for doc in docs_ts]
del id2word

logging.info(f'Final number of documents (train): {len(words_tr)}')
logging.info(f'Final number of documents (test): {len(words_ts)}')

train_documents = {
    "split": words_tr,
    "joined": list(map(lambda x: " ".join(x), words_tr)),
}
documents_path = OUTPUT_PATH + "/train_documents.json"
logging.info(f'Saving train documents file with {len(train_documents["split"])} documents (JSON): {documents_path}')
os.makedirs(os.path.dirname(documents_path), exist_ok=True)
json.dump(train_documents, open(documents_path, 'w'))
logging.info(f'Train documents file saved (JSON): {documents_path}')

test_documents = {
    "split": words_ts,
    "joined": list(map(lambda x: " ".join(x), words_ts)),
}
documents_path = OUTPUT_PATH + "/test_documents.json"
logging.info(f'Saving joined test documents file with {len(test_documents["split"])} documents (JSON): {documents_path}')
os.makedirs(os.path.dirname(documents_path), exist_ok=True)
json.dump(test_documents, open(documents_path, 'w'))
logging.info(f'Test documents file saved (JSON): {documents_path}')
del test_documents

logging.info("Creating word dictionary for entire corpus...")
path_save = OUTPUT_PATH + '/word_dictionary.gdict'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(dictionary, path_save, compress=8)
logging.info(f'Word dictionary created and saved to "{path_save}"')
del dictionary
del words_ts

# Getting lists of words and doc_indices
logging.info('(ETM) Creating lists of words...')

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]


words_tr = create_list_words(docs_tr)

# Get doc indices
logging.info('(ETM) Getting doc indices...')

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]


doc_indices_tr = create_doc_indices(docs_tr)
logging.info('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
doc_indices_ts = create_doc_indices(docs_ts)
logging.info('  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))

# Number of documents in each set
n_docs_tr = len(docs_tr)
n_docs_ts = len(docs_ts)

# Remove unused variables
del docs_tr
del docs_ts

# Create bow representation
logging.info('(ETM) Creating bow representation...')

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()


bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))

del words_tr
del doc_indices_tr
del doc_indices_ts

logging.info('(ETM) Splitting bow intro token/value pairs and saving...')

def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    return indices, counts


bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
del bow_tr

etm_training_dataset = {
    "tokens": _to_numpy_array(bow_tr_tokens), 
    "counts": _to_numpy_array(bow_tr_counts),
}

logging.info("Creating ETM vocabulary file...")
path_save = OUTPUT_PATH + '/etm_vocabulary.vocab'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(vocab, path_save, compress=8)
logging.info(f'ETM vocabulary saved to "{path_save}"')

logging.info("Creating ETM training dataset file...")
path_save = OUTPUT_PATH + '/etm_training_dataset.dataset'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(etm_training_dataset, path_save, compress=8)
logging.info(f'ETM training dataset saved to "{path_save}"')
del bow_tr_tokens
del bow_tr_counts
del etm_training_dataset

logging.info("Creating ETM embeddings file with words in vocabulary")
path_save = OUTPUT_PATH +  '/etm_w2v_embeddings.w2v'
optimize_embeddings(
    vocab,
    word_lemma_maps["lemma_word"],
    args.embeddings,
    path_save,
    args.n_dim,
    logging,
)
logging.info("ETM embeddings file with words in vocabulary created")
del word_lemma_maps
del vocab

logging.info("Creating CTM training dataset file...")
simple_preprocessing = WhiteSpacePreprocessing(train_documents["joined"], "portuguese")
del train_documents

preprocessed_documents_for_bow, unpreprocessed_corpus_for_contextual, vocab = simple_preprocessing.preprocess()
logging.info(f'CTM: preprocessed_documents_for_bow = {len(preprocessed_documents_for_bow)}')
logging.info(f'CTM: unpreprocessed_corpus_for_contextual = {len(unpreprocessed_corpus_for_contextual)}')
logging.info(f'CTM: vocab = {len(vocab)}')

data_preparation = TopicModelDataPreparation("distiluse-base-multilingual-cased")
ctm_training_dataset = data_preparation.create_training_set(unpreprocessed_corpus_for_contextual, preprocessed_documents_for_bow)

path_save = OUTPUT_PATH + '/ctm_data_preparation.obj'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(data_preparation, path_save, compress=8)
logging.info(f'CTM data preparation instance saved to "{path_save}"')

path_save = OUTPUT_PATH + '/ctm_training_dataset.dataset'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(ctm_training_dataset, path_save, compress=8)
logging.info(f'CTM training dataset saved to "{path_save}"')

logging.info('\nDatasets prepared for training')

end = time.time()

logging.info(f'\n\nElapsed execution time for preparation: {end-start}s')
