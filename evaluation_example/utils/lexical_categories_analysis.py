from collections import Counter
import liwc
import re
from empath import Empath


def get_liwc_categories_for_topics(model, liwc_dictionary_path, normalize=False):
    parse, category_names = liwc.load_token_parser(liwc_dictionary_path)

    topics_categories = []
    for topic in model["topics"]:
        categories = [category for token in topic for category in parse(token)]
        counts = Counter(categories)
        no_of_words = len(topic)
        counts = counts if normalize is False else dict([(key, counts[key]/no_of_words) for key in counts.keys()])
        topics_categories.append(counts)

    return topics_categories


def get_empath_categories_for_topics(model, normalize=False):
    lex = Empath()

    topics_categories = []
    for topic in model["topics"]:
        result = lex.analyze(" ".join(model["topics"]), normalize=normalize)
        counts = dict([(key, result[key]) for key in result.keys() if result[key] > 0])
        topics_categories.append(counts)
    
    return topics_categories


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
