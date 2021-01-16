from utils.preprocessor import Preprocessor
from utils.parse import str2bool
import pandas as pd
import numpy as np
import sys
import json
import re
import os
from nltk.tokenize.util import is_cjk
import argparse


OUTPUT_PATH = "datasets/processed"
FILE_EXTENSION = ".json"
WRITE_MODE = "w"
READ_MODE = "r"


UNDESIRED_WORDS_LIST = ["https", "http", "www", "removed", "deleted"]


def get_filename(path):
    lowercase_path = path.lower()

    filename_with_extension  = lowercase_path.split("/")[-1]

    filename_without_extension = re.sub(FILE_EXTENSION, "", filename_with_extension)

    return filename_without_extension


def remove_bots_posts(df, bots=["AutoModerator", "RemindMeBot", "WikiTextBot", "youtubefactsbot", "RedditNiobioBot", "NemLiNemLereiBot"]):
    """Verifies if given word has japanese chars.

    Parameters:
    df (pandas.DataFrame): dataset to remove bots' posts

    bots (list) - optional: list of str with bot names to look for and remove respective posts

    Returns:
    Pandas.DataFrame: dataset without posts made by bots
    
    """
    df_without_bot_posts = df[~df.author.isin(bots)]

    return df_without_bot_posts


def is_jp_word(word):
    """Verifies if given word has japanese chars.

    Parameters:
    word (str): word to evaluate

    Returns:
    bool: wheter or not japanese chars exists in the string
    
    """
    return any([ is_cjk(char) for char in word ])


def has_undesired_word(text):
    return any(list(map(lambda word: (is_jp_word(word) or word in UNDESIRED_WORDS_LIST), text)))


def remove_undesired_words(df):
    return df[df['body'].map(lambda text: has_undesired_word(text)) == False]



parser = argparse.ArgumentParser(description='Preprocessor for dataset preprocessing: stopwords removal, lemmatization, duplicate records filtering and etcetera.')

parser.add_argument('--datasetFile', type=str, help='dataset file', required=True)
parser.add_argument('--datasetName', type=str, help='dataset name', required=True)
parser.add_argument('--datasetFolder', type=str, help='dataset folder', required=True)
parser.add_argument('--stopwordsFile', type=str, help='additional stop-words file', required=False)
parser.add_argument('--field', type=str, help='field to be processed', required=True)
parser.add_argument('--lang', type=str, help='dataset language: string "en" or "pt"', required=True)
parser.add_argument('--lemmatize', type=str2bool, help='should lemmatize data?', required=False, default=False)
parser.add_argument('--removeStopwords', type=str2bool, help='should remove stopwords?', required=False, default=False)
parser.add_argument('--removePos', type=str2bool, help='should remove POS categories?', required=False, default=False)
parser.add_argument('--desiredPos', nargs='+', help='part-of-speech categories to keep. These are simple Spacy POS categories', required=False, default=['NOUN'])

args = parser.parse_args()

original_data_path = args.datasetFile
dataset_name = args.datasetName
dataset_folder = args.datasetFolder
stopwords_file = args.stopwordsFile
field_of_interest = args.field
lang = args.lang
lemmatize_activated = args.lemmatize
remove_pos = args.removePos
remove_stopwords = args.removeStopwords
posCategories = args.desiredPos

print("Path: ", original_data_path)
if stopwords_file != None: print("Additional stopwords file: ", stopwords_file)
print("Field: ", field_of_interest)
print("Dataset: ", dataset_name)
print("Folder: ", dataset_folder)
print("Language: ", lang)
print("Is to lemmatize data? ", lemmatize_activated)
print("Is to remove stopwords? ", remove_stopwords)
print("Is to remove POS categories? ", remove_pos)
print("POS categories to keep: ", posCategories)

data_string = json.load(open(original_data_path, READ_MODE))

original_data_frame = pd.DataFrame.from_dict(data_string)

print(original_data_frame.head())

data = np.array(original_data_frame[field_of_interest], dtype = 'object')

processor = Preprocessor(
    posCategories, 
    lang, 
    lemmatize_activated=lemmatize_activated, 
    remove_pos=remove_pos, 
    remove_stopwords=remove_stopwords
)

processed_data = processor.preprocess(data, stopwords_file)

print("Size of data after preprocessing: ", len(processed_data))

df_after_preprocessing = original_data_frame.assign(body=processed_data)

df_after_preprocessing = df_after_preprocessing[df_after_preprocessing['body'].map(lambda field: len(field)) > 0]

print(f'Row count after removal of rows with empty "{field_of_interest}" fields: {len(df_after_preprocessing)}')

#output_filepath = OUTPUT_PATH + get_filename(original_data_path) + "[processed]" + FILE_EXTENSION
output_filepath = "/".join([OUTPUT_PATH, dataset_folder, dataset_name]) + "[processed]" + FILE_EXTENSION

os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

json.dump(df_after_preprocessing.to_dict(orient='records'), open(output_filepath, WRITE_MODE))

print("Data dumped to ", output_filepath)
