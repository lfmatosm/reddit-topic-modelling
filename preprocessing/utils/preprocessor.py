from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
import re
import emoji
import json
import os


class MemoryFriendlyFileIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


class MemoryFriendlyJSONFileIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield json.loads(line)


class Preprocessor:
    """Makes basic, generic preprocessing on documents corpus.

    Parameters:
    pos_categories (list): part-of-speech categories to maintain

    language ('en' or 'pt') - optional: corpus language

    lemmatize_activated (bool) - optional: if should lemmatize words
    """
    def __init__(
        self, pos_categories, 
        logger=None,
        language="en", 
        lemmatize_activated=True, 
        remove_pos=True,
        remove_stopwords=True
    ):
        self.__logger = logger if logger is not None else print
        self.__lemmatize_activated = lemmatize_activated
        self.__remove_pos_activated = remove_pos
        self.__remove_stopwords_activated = remove_stopwords
        self.__pos_categories = pos_categories

        self.__nlp = spacy.load("pt_core_news_sm") if (language == "pt") else spacy.load("en_core_web_sm")

        raw_stopwords = stopwords.words("portuguese") if (language == "pt") else stopwords.words("english")
        deaccent_stopwords = simple_preprocess(" ".join(raw_stopwords), deacc=True, min_len=1, max_len=100)
        self.__stop_words = list(set(raw_stopwords+deaccent_stopwords))


    #Removes newline chars from each document
    def remove_newlines_and_single_quotes(self, texts):
        remove_newlines = lambda text: re.sub(r'\s+', ' ', text)

        remove_single_quotes = lambda text: re.sub("\'", "", text)

        without_newlines = map(remove_newlines, texts)

        return map(remove_single_quotes, without_newlines)


    #Removes emojis
    def remove_emojis(self, texts):
        return list(map(lambda text: emoji.get_emoji_regexp().sub(u'', text), list(texts)))


    #Removes documents with size of less than n words
    def filter_documents_with_less_than(self, tokenized_documents, min_words=5):
        return list(filter(lambda tokenized_text: len(tokenized_text) > min_words, tokenized_documents))

    
    #Removes small words from documents
    def remove_small_words(self, tokenized_documents, min_length=3):
        return [list(filter(lambda word: len(word) > min_length, document)) for document in tokenized_documents]


    #Removes stopwords
    def remove_stopwords(self, texts, additional_stopwords=[]):
        stopwords = self.__stop_words + additional_stopwords
        self.__logger(f'Using stoplist of length: {len(stopwords)}')

        count = 0

        documents_without_stopwords = []
        for document in texts:
            document_without_stopwords = []
            for word in document:
                if word not in stopwords:
                    document_without_stopwords.append(word)
                else:
                    count += 1
            documents_without_stopwords.append(document_without_stopwords)

        self.__logger(f'No. of stopwords removed: {count}')
        return documents_without_stopwords


    def save_to_temp_file(self, documents):
        path = 'temp.tmp'
        with open(path, 'w') as file:
            for document in documents:
                file.write(f'{" ".join(document)}\n')
        return path


    def load_from_temp_file(self, path):
        iterator = MemoryFriendlyFileIterator(path)

        return [line for line in iterator]

    
    #Transforms each word into its base form. e.g. 'fazendo' becomes 'fazer'
    def lemmatize(self, path):
        output = 'temp2.tmp'
        documents = MemoryFriendlyFileIterator(path)

        with open(output, 'w') as file:
            for document in documents:
                doc_string = self.__nlp(" ".join(document))

                document_tokens = [{
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                } for token in doc_string]
                file.write(f'{json.dumps(document_tokens)}\n')

        return output
    
    def get_lemmas(self, path):
        output = 'temp3.tmp'
        documents = MemoryFriendlyJSONFileIterator(path)

        with open(output, 'w') as file:
            for document in documents:
                tokens = [token['lemma'] for token in document]
                file.write(f'{" ".join(tokens)}\n')

        return output

    
    def filter_part_of_speech_tags(self, path, categories):
        output = 'temp3.tmp'
        documents = MemoryFriendlyJSONFileIterator(path)

        with open(output, 'w') as file:
            for document in documents:
                tokens = [token['lemma'] for token in document if token['pos'] in categories]
                file.write(f'{" ".join(tokens)}\n')

        return output


    def preprocess(self, data, stopwords_file_path=None):
        """Realizes preprocessing on the given data object. Removes special characters 
        and accentuations, breaks sentences into tokens, filter part-of-speech (POS) 
        word categories, lemmatkze and remove stopwords from the corpus.

        Parameters:
        data (numpy.array): dataset to preprocess

        stopwords_file_path (str) - optional: path of file with additional stopwords. 
        If not given, stopwords from "nltk" are used by default

        Returns:
        two-dimensional array: dataset after preprocessing
        
        """
        data_without_newlines = self.remove_newlines_and_single_quotes(data)

        self.__logger("Newlines and single-quotes removed from documents.")

        data_without_emojis = self.remove_emojis(data_without_newlines)

        self.__logger("Emojis removed from documents.")

        #Breaks each document into a list of words
        tokenize = lambda texts: [(yield simple_preprocess(text, deacc=True, min_len=1, max_len=30)) \
            for text in texts]

        tokenized_data = tokenize(data_without_emojis)
        del data_without_emojis

        self.__logger("Tokenized documents.")

        path = self.save_to_temp_file(tokenized_data)
        del tokenized_data
        self.__logger("Saved tokenized documents to temp file for further processing...")

        if self.__lemmatize_activated:
            path = self.lemmatize(path)

            self.__logger("Lemmatized documents.")

        if self.__remove_pos_activated:
            path = self.filter_part_of_speech_tags(path, self.__pos_categories)

            self.__logger(f'{", ".join(self.__pos_categories)} POS categories of tokens kept and lemmatized.')
        elif self.__lemmatize_activated:
            path = self.get_lemmas(path)

            self.__logger("Token lemmas maintained on documents")
        
        preprocessed_data = self.load_from_temp_file(path)
        self.__logger("Read processed documents from temp file")

        stopwords = None

        if self.__remove_stopwords_activated:
            additional_stopwords = open(stopwords_file_path, "r").read().split(",") if stopwords_file_path is not None else []

            preprocessed_data = self.remove_stopwords(preprocessed_data, additional_stopwords)
            print(f'stopwords removed = {preprocessed_data}')

            self.__logger("Stopwords removed.")
        else:
            stopwords = self.__stop_words

        for path in ['temp.tmp', 'temp2.tmp', 'temp3.tmp']:
            if os.path.exists(path):
                os.remove(path)
        self.__logger("Removed temporary files.")

        return preprocessed_data, stopwords
