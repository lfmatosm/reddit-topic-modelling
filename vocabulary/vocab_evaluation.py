from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
import argparse
import json
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import os
import joblib
import sys


OUTPUT_PATH = "./images/"
DICTIONARY_FOLDER = "dictionary"
# if 21: [0.   0.05 0.1  0.15 0.2  0.25 0.3  0.35 0.4  0.45 0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95 1.]
# if 11: [0. 0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.]
N_BINS = 11
FILE_EXTENSION = ".pdf"

# logger = LoggerFactory.get_instance().getLogger('vocab_evaluation')

def normalize(arr, size):
    return np.array(arr) / size


def get_range_to_plot(frequencies):
    end = len(frequencies) - 1
    start = 0

    for i in range(end, 0, -1):
        if int(frequencies[i]) == 0:
            end = i
        else:
            break
    
    for i in range(start, end, 1):
        if int(frequencies[i]) == 0:
            start = i
        else:
            break

    return start, end


def get_document_frequency_counts(documents, min_df_to_analyse=0.0, max_df_to_analyse=1.0):
    total_documents = len(documents)
    print(f'Total docs: {total_documents}')

    dictionary = Dictionary(documents)

    print(f'Distinct tokens: {len(dictionary.token2id)}')
    print(f'Docs processed: {dictionary.num_docs}')
    print(f'Total words processed: {dictionary.num_pos}\n\n')

    document_freqs = dictionary.dfs

    print(f'Doc-freqs length: {len(document_freqs)}')

    min_range = int(min_df_to_analyse*total_documents)
    max_range = int(max_df_to_analyse*total_documents)

    print(f'Look for token counts in the range: ({min_range}, {max_range})')
    interval_markers = np.linspace(min_range, max_range, N_BINS)
    intervals = [(interval_markers[i-1]+1, interval_markers[i]) for i in range(1, len(interval_markers))]

    print(f'Analysis intervals: {intervals}')

    document_freqs_by_interval = [0] * len(intervals)
    for value in document_freqs.values():
        for idx, (begin, end) in enumerate(intervals):
            if begin <= value and value <= end:
                document_freqs_by_interval[idx] += 1
                break

    print(f'Documents-frequencies by interval: {document_freqs_by_interval}')

    bins = list(map(lambda x: x[0], intervals))
    bins.append(intervals[-1][1])
    bins = normalize(bins, total_documents)

    start, end = get_range_to_plot(document_freqs_by_interval)

    return document_freqs_by_interval, bins, start, end, dictionary


def plot_aggregated_histogram(weights, x, nbins, title, lang):
    fig, ax = plt.subplots(tight_layout=True)
    test = ax.hist(x, bins=N_BINS, weights=weights)
    ax.set(xlabel='Fração de documentos do corpus', ylabel="Número de tokens únicos")

    filename = os.path.join(OUTPUT_PATH, lang, f'{title}_{FILE_EXTENSION}')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, bbox_inches='tight')

def plot_histogram(points, bins, title):
    fig, ax = plt.subplots(tight_layout=True)
    test = ax.hist(points, bins=None)
    ax.set(xlabel='Fração de documentos do corpus', ylabel="Frequência em documentos")

    filename = os.path.join(OUTPUT_PATH, f'{title}{FILE_EXTENSION}')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, bbox_inches='tight')

    # plt.show()


def remove_stopwords(documents, stopwords):
    count = 0

    documents_without_stopwords = []
    for document in documents:
        document_without_stopwords = []
        for word in document:
            if word not in stopwords:
                document_without_stopwords.append(word)
            else:
                count += 1
        documents_without_stopwords.append(document_without_stopwords)

    print(f'No. of stopwords removed: {count}')
    return documents_without_stopwords


def load_stopwords(stopwords_file):
    stopwords = json.load(open(args.stopwords))
    stopwords = simple_preprocess(" ".join(stopwords), deacc=True, min_len=1, max_len=100)
    return stopwords


def get_min_df_value_from_fraction(df_fraction, total_documents):
    return int(total_documents*df_fraction)

parser = argparse.ArgumentParser(description='Analyses vocabulary composition')
parser.add_argument('--dataset', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--lang', type=str, help='dataset language', required=False, default='en')
parser.add_argument('--min_df_to_analyse', type=float, help='min DF to count tokens for analysis. Use only if you know what are you doing', required=False, default=0.0)
parser.add_argument('--max_df_to_analyse', type=float, help='max DF to count tokens for analysis. Use only if you know what are you doing', required=False, default=1.0)
# parser.add_argument('--stopwords', type=str, help='stopwords path. A JSON file', required=False, default=None)
args = parser.parse_args()

raw_documents = json.load(open(args.dataset))
documents = [document["body"] for document in raw_documents]
total_documents = len(documents)

words_document_frequencies, bins, start, end, dictionary = get_document_frequency_counts(
    documents,
    args.min_df_to_analyse,
    args.max_df_to_analyse,
)

x = bins[start:end]
weights = words_document_frequencies[start:end]
nbins = len(range(start, end))
print(f'Relevant marks: {x}')
print(f'Relevant frequencies for each bin: {weights}')
print(f'No. of bins: {nbins}')

plot_aggregated_histogram(
    weights, 
    x, 
    nbins,
    f'histogram_{args.lang}_{start}_to_{end}',
    args.lang,
)
print(f'Histogram generated and saved\n\n')

stop = int(input('Finish vocabulary analysis: (0 for no, 1 for yes): ')) > 0
if stop:
    sys.exit(0)

original_min_df = 0
min_df = 0
max_df = 1.0

filter_by_min_df = int(input('Do you want to filter tokens by minimum DF? ')) > 0
if filter_by_min_df:
    original_min_df = float(input('Insert the minimum DF to filter (range: 0-1): '))
    min_df = get_min_df_value_from_fraction(original_min_df, total_documents)

filter_by_max_df = int(input('Do you want to filter tokens by maximum DF? ')) > 0
if filter_by_max_df:
    max_df = float(input('Insert the maximum DF to filter (range: 0-1): '))

print(f'\n\nUsing\n\tNo below: {min_df} documents\n\tNo above: {max_df} of total corpus')
dictionary.filter_extremes(
    no_below=min_df,
    no_above=max_df,
)
print(f'Distinct tokens after filtering: {len(dictionary.token2id)}')

path = os.path.join(
    OUTPUT_PATH, 
    DICTIONARY_FOLDER, 
    f'dictionary_{args.lang}_min_df_{original_min_df}_max_df_{max_df}.gdict'
)
os.makedirs(os.path.dirname(path), exist_ok=True)
joblib.dump(
    dictionary,
    path,
    compress=8,
)
print(f'Saved filtered dictionary instance to {path}')

# if args.stopwords is not None:
#     stopwords = load_stopwords(args.stopwords)
#     documents_without_stopwords = remove_stopwords(documents, stopwords)

#     words_document_frequencies, bins, dictionary = get_document_frequency_counts(documents_without_stopwords)
#     plot_histogram(words_document_frequencies, bins, "histogram_with_stopwords_removal")
