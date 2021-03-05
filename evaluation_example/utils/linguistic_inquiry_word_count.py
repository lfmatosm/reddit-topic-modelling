from collections import Counter
import liwc
import re


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
