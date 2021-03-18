from sklearn.feature_extraction.text import CountVectorizer
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

class MemoryFriendlyFileIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


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

def clean_w2v_embedding_of_words_not_in_vocabulary(
    vocabulary, 
    lemma_word_mapping,
    embedding_file,
    output_embedding_path, 
    n_dim
):
    temp_file = output_embedding_path + "_TMP"

    print("Generating smaller W2V embedding based on vocabulary words...")

    iterator = MemoryFriendlyFileIterator(embedding_file)

    words_not_found = [word for word in vocabulary]

    nlines = 0
    with open(temp_file, 'w') as f:
        for line in iterator:
            word = line[0]
            if word in vocabulary:
                words_not_found.remove(word)
                vector = np.array(line[1:]).astype(np.float)
                vec_str = [f'{val}' for val in vector]
                f.write(f'{word} {" ".join(vec_str)}\n')
                nlines += 1
    
    del iterator
    print(f'Smaller W2V embedding created with vocabulary words found: {nlines}/{len(vocabulary)}')
    iterator = MemoryFriendlyFileIterator(embedding_file)

    if len(words_not_found) > 0:
        with open(temp_file, 'a') as f:
            for line in iterator:
                word = line[0]
                words_added_now = []
                for word_not_found in words_not_found:
                    if word_not_found not in words_added_now and word in lemma_word_mapping[word_not_found]:
                        words_added_now.append(word_not_found)
                        vector = np.array(line[1:]).astype(np.float)
                        vec_str = [f'{val}' for val in vector]
                        f.write(f'{word_not_found} {" ".join(vec_str)}\n')
                        print(f'Vector for "{word_not_found}" not found on W2V embeddings, replacing it with "{word}" vector')
                        nlines += 1
                words_not_found = list(set(words_not_found)-set(words_added_now))

    del iterator
    del words_not_found
    iterator = MemoryFriendlyFileIterator(temp_file)

    with open(output_embedding_path, 'w') as f:
        f.write(f'{nlines} {n_dim}\n')
        for line in iterator:
            f.write(f'{" ".join(line)}\n')

    del iterator

    if os.path.exists(temp_file):
        os.remove(temp_file)    
    del temp_file
    print(f'Total words added to smaller W2V embedding: {nlines}/{len(vocabulary)}')
    del nlines

    print("Generating optimized Gensim W2V embedding file from smaller W2V embedding...")
    embeddings = KeyedVectors.load_word2vec_format(
        output_embedding_path, 
        binary=False
    )
    gensim_embedding_output_path = output_embedding_path.replace('.txt', '.w2v')
    os.makedirs(os.path.dirname(gensim_embedding_output_path), exist_ok=True)
    embeddings.save(gensim_embedding_output_path)
    del embeddings
    print(f'Generated optimized Gensim W2V embedding file at "{gensim_embedding_output_path}"')
    del gensim_embedding_output_path


parser = argparse.ArgumentParser(description='Prepares training/testing resources for LDA/CTM/ETM training scripts')
parser.add_argument('--dataset', type=str, help='dataset path', required=True)
parser.add_argument('--word_lemma_maps', type=str, help='word-lemma mappings path', required=False, default=None)
# parser.add_argument('--dictionary', type=str, help='dictionary path', required=True)
parser.add_argument('--stopwords', type=str, help='stopwords path', required=False, default=None)
parser.add_argument('--embeddings', type=str, help='embeddings path', required=True)
parser.add_argument('--n_dim', type=int, help='embeddings size', required=False, default=300)
parser.add_argument('--train_size', type=float, help='train size', required=False, default=1.0)
parser.add_argument('--dataset_name', type=str, help='dataset path', default='training_data', required=False)
parser.add_argument('--min_df', default=None, type=float, help='minimum document frequency for document vectorizer', required=False)
parser.add_argument('--max_df', default=None, type=float, help='minimum document frequency for document vectorizer', required=False)
args = parser.parse_args()

min_df=args.min_df if args.min_df is not None else 1
max_df=args.max_df if args.max_df is not None else 1.0
dataset=args.dataset
dataset_name = args.dataset_name
train_size = float(args.train_size)

# ADDITIONAL_STOPWORDS = ["http", "https", "watch", "comment", "comments"]
ADDITIONAL_STOPWORDS = ["http", "https"]

print(f'dataset: {dataset}\ndataset_name: {dataset_name}\ntrain_size: {train_size}\nstopwords: {args.stopwords}\nmin_df: {min_df}\nmax_df: {max_df}\n')

OUTPUT_PATH = f'./resources/{dataset_name}'

print("Loading dataset file...")
documents = json.load(open(dataset, "r"))
joined_documents = [ " ".join(document["body"]) for document in documents ]
print(f'Dataset length: {len(joined_documents)}')
del documents

# print("Loading Gensim dictionary...")
# dictionary = joblib.load(args.dictionary)
# print(f'No. of tokens on dictionary: {len(dictionary.token2id)}')

word_lemma_maps = None
if args.word_lemma_maps is not None:
    print("Loading word-lemma mappings...")
    word_lemma_maps = json.load(open(args.word_lemma_maps, "r"))
    print(f'Word-lemma and inverse mappings loaded with {len(word_lemma_maps)} entries')

stopwords = None
if args.stopwords is not None:
    print("Loading stopwords...")
    stopwords = json.load(open(args.stopwords, "r"))
    print(f'Stopwords loaded with {len(stopwords)} entries')

print("Counting word frequencies...")
vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stopwords)
print("Word frequencies computed")
del min_df
del max_df
del stopwords

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
del vectorized_documents

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
trSize = int(np.floor(train_size*num_docs))
tsSize = int(num_docs - trSize)
del cvz
idx_permute = np.random.permutation(num_docs).astype(int)

print('Vocabulary length: {}'.format(len(vocab)))

docs_tr = [[word2id[w] for w in docs[idx_permute[idx_d]] if w in word2id] for idx_d in range(trSize)]
docs_ts = [[word2id[w] for w in docs[idx_permute[idx_d]] if w in word2id] for idx_d in range(tsSize)]
del docs

print('Number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
print('Number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))

# Remove empty documents
print('Removing empty documents...')

docs_tr = remove_empty(docs_tr)
docs_ts = remove_empty(docs_ts)

# Obtains the training, test and validation datasets as word lists
words_tr = [[id2word[w] for w in doc] for doc in docs_tr]
words_ts = [[id2word[w] for w in doc] for doc in docs_ts]

train_documents = {
    "split": words_tr,
    "joined": list(map(lambda x: " ".join(x), words_tr)),
}
documents_path = OUTPUT_PATH + "/train_documents.json"
print(f'Saving train documents file with {len(train_documents["split"])} documents (JSON): {documents_path}')
os.makedirs(os.path.dirname(documents_path), exist_ok=True)
json.dump(train_documents, open(documents_path, 'w'))
print(f'Train documents file saved (JSON): {documents_path}')

test_documents = {
    "split": words_ts,
    "joined": list(map(lambda x: " ".join(x), words_ts)),
}
documents_path = OUTPUT_PATH + "/test_documents.json"
print(f'Saving joined test documents file with {len(test_documents["split"])} documents (JSON): {documents_path}')
os.makedirs(os.path.dirname(documents_path), exist_ok=True)
json.dump(test_documents, open(documents_path, 'w'))
print(f'Test documents file saved (JSON): {documents_path}')
del test_documents

print("Creating word dictionary for entire corpus...")
dictionary = create_dictionary(words_tr + words_ts)
path_save = OUTPUT_PATH + '/word_dictionary.gdict'
os.makedirs(os.path.dirname(path_save), exist_ok=True)
joblib.dump(dictionary, path_save, compress=7)
print(f'Word dictionary created and saved to "{path_save}"')
del dictionary
del words_ts

# Getting lists of words and doc_indices
print('(ETM) Creating lists of words...')

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]

words_tr = create_list_words(docs_tr)
# words_ts = create_list_words(docs_ts)

print('  len(words_tr): ', len(words_tr))
# print('  len(words_ts): ', len(words_ts))

# Get doc indices
print('(ETM) Getting doc indices...')

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

doc_indices_tr = create_doc_indices(docs_tr)
print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
doc_indices_ts = create_doc_indices(docs_ts)
print('  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))

# Number of documents in each set
n_docs_tr = len(docs_tr)
n_docs_ts = len(docs_ts)

# Remove unused variables
del docs_tr
del docs_ts

# Create bow representation
print('(ETM) Creating bow representation...')

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
# bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))

del words_tr
del doc_indices_tr
# del words_ts
del doc_indices_ts

# Split bow intro token/value pairs
print('(ETM) Splitting bow intro token/value pairs and saving...')

def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    return indices, counts

bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
# bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
del bow_tr
# del bow_ts

etm_training_dataset = {
    "tokens": _to_numpy_array(bow_tr_tokens), 
    "counts": _to_numpy_array(bow_tr_counts),
}

# etm_testing_dataset = {
#     "tokens": _to_numpy_array(bow_ts_tokens), 
#     "counts": _to_numpy_array(bow_ts_counts),
# }

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
del bow_tr_tokens
del bow_tr_counts
del etm_training_dataset

# print("Creating ETM testing dataset file...")
# path_save = OUTPUT_PATH + '/etm_testing_dataset.dataset'
# os.makedirs(os.path.dirname(path_save), exist_ok=True)
# joblib.dump(etm_testing_dataset, path_save, compress=7)
# print(f'ETM testing dataset saved to "{path_save}"')
# del bow_ts_tokens
# del bow_ts_counts
# del etm_testing_dataset

print("Creating ETM embeddings file with words in vocabulary")
path_save = OUTPUT_PATH +  '/etm_w2v_embeddings.txt'
clean_w2v_embedding_of_words_not_in_vocabulary(
    vocab,
    word_lemma_maps["lemma_word"],
    args.embeddings,
    path_save,
    args.n_dim
)
print("ETM embeddings file with words in vocabulary created")
del word_lemma_maps
del vocab

print("Creating CTM training dataset file...")
simple_preprocessing = WhiteSpacePreprocessing(train_documents["joined"], "portuguese")
del train_documents

preprocessed_documents_for_bow, unpreprocessed_corpus_for_contextual, vocab = simple_preprocessing.preprocess()
print(f'CTM: preprocessed_documents_for_bow = {len(preprocessed_documents_for_bow)}')
print(f'CTM: unpreprocessed_corpus_for_contextual = {len(unpreprocessed_corpus_for_contextual)}')
print(f'CTM: vocab = {len(vocab)}')

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
