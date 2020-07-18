from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
import re


class Preprocessor:
    """Makes basic, generic preprocessing on documents corpus.

    Parameters:
    pos_categories (list): part-of-speech categories to maintain

    language ('en' or 'pt') - optional: corpus language

    lemmatize_activated (bool) - optional: if should lemmatize words
    """
    def __init__(self, pos_categories, language="en", lemmatize_activated=True):
        self.__lemmatize_activated = lemmatize_activated
        self.__pos_categories = pos_categories

        self.__nlp = spacy.load("pt_core_news_sm") if (language == "pt") else spacy.load("en")
        self.__stop_words = stopwords.words("portuguese") if (language == "pt") else stopwords.words("english")


    #Removes newline chars from each document
    def remove_newlines_and_single_quotes(self, texts):
        remove_newlines = lambda text: re.sub(r'\s+', ' ', text)

        remove_single_quotes = lambda text: re.sub("\'", "", text)

        without_newlines = map(remove_newlines, texts)

        return map(remove_single_quotes, without_newlines)


    #Removes documents with size of less than n words
    def filter_documents_with_less_than(self, tokenized_documents, min_words=5):
        return list(filter(lambda tokenized_text: len(tokenized_text) > min_words, tokenized_documents))

    
    #Removes small words from documents
    def remove_small_words(self, tokenized_documents, min_length=3):
        return [list(filter(lambda word: len(word) > min_length, document)) for document in tokenized_documents]


    #Removes stopwords
    def remove_stopwords(self, texts, additional_stopwords=None):
        documents = [[word for word in doc if word not in self.__stop_words] for doc in texts]

        return documents if additional_stopwords == None else [[word for word in doc if word not in additional_stopwords] for doc in documents]


    #Transforms each word into its base form. e.g. 'fazendo' becomes 'fazer'
    def lemmatize(self, documents):
        return [[ token.lemma_ for token in document ] for document in documents]
    

    #For each word, produces a (word, pos, lemma) pair where pos is the part-of-speech category of the given word/token
    def filter_part_of_speech_tags(self, documents, categories):
        tokens_with_pos = []

        for document in documents:
            doc_string = self.__nlp(" ".join(document))
            tokens_with_pos.append([ token.lemma_ for token in doc_string if token.pos_ in categories])

        return tokens_with_pos
    

    #Removes POS categories not explicitly specified to be kept on the corpus.
    def filter_part_of_speech_categories(self, documents, categories):
        return [[ token for token, pos in document if pos in categories ] for document in documents]


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

        print("Newlines and single-quotes removed from documents")

        #Breaks each document into a list of words
        tokenize = lambda texts: [(yield simple_preprocess(text, deacc=True, min_len=1)) for text in texts]

        tokenized_data = tokenize(data_without_newlines)

        print("Tokenized documents.")

        tokens_with_pos = self.filter_part_of_speech_tags(tokenized_data, self.__pos_categories)

        print(f'{", ".join(self.__pos_categories)} POS categories of tokens kept and lemmatized.')

        additional_stopwords = open(stopwords_file_path, "r").read().split(",") if stopwords_file_path != None else None

        data_without_stopwords = self.remove_stopwords(tokens_with_pos, additional_stopwords)

        print("Stopwords removed.")

        return self.remove_small_words(data_without_stopwords)
