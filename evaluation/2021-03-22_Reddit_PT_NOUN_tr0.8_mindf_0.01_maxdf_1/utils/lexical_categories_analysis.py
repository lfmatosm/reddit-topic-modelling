from collections import Counter
import liwc
import re
from empath import Empath
import itertools


def get_all_topic_words(model):
    return list(itertools.chain.from_iterable(model["topics"]))


def sort_counts(counts):
    counts_list = list((k, v) for k, v in counts.items())
    counts_list.sort(key=lambda x: x[1])
    counts_list.reverse()
    return counts_list


def get_raw_liwc_categories_for_topics(model, liwc_dictionary_path):
    parse, category_names = liwc.load_token_parser(liwc_dictionary_path)
    categories = []
    for topic in model["topics"]:
        word_categories = {}
        for word in topic:
            word_categories[word] = [category for category in parse(word)]
        categories.append(word_categories)
    return categories


def get_liwc_categories_for_topics(model, liwc_dictionary_path, normalize=False):
    parse, category_names = liwc.load_token_parser(liwc_dictionary_path)
    topic_words = get_all_topic_words(model)
    categories = [category for token in topic_words for category in parse(token)]
    counts = Counter(categories)
    no_of_words = len(topic_words)
    return sort_counts(counts if normalize is False else dict([(key, counts[key]/no_of_words) for key in counts.keys()]))


def get_raw_empath_categories_for_topics(model):
    lex = Empath()
    categories = []
    for topic in model["topics"]:
        word_categories = {}
        for word in topic:
            result = lex.analyze(word)
            word_categories[word] = [key for key in result.keys() if result[key] > 0]
        categories.append(word_categories)
    return categories


def get_empath_categories_for_topics(model, normalize=False):
    lex = Empath()
    topic_words = get_all_topic_words(model)
    result = lex.analyze(" ".join(topic_words), normalize=normalize)
    return sort_counts(dict([(key, result[key]) for key in result.keys() if result[key] > 0]))


def get_categories_for_text(text, liwc_dictionary_path):
    parse, category_names = liwc.load_token_parser(liwc_dictionary_path)

    def tokenize(text):
        # you may want to use a smarter tokenizer
        for match in re.finditer(r'\w+', text, re.UNICODE):
            yield match.group(0)

    tokens = tokenize(text.lower())

    categories = [category for token in tokens for category in parse(token)]
    counts = Counter(categories)
    
    return categories, counts


def get_categories_for_word(word, liwc_dictionary_path):
    parse, category_names = liwc.load_token_parser(liwc_dictionary_path)
    return [category for category in parse(word)]
