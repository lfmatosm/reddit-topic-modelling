import os
import joblib
import argparse

MAX_WORDS_PER_TOPIC = 5
START_COMMENT = "% Inicio de itemize\n\n"
END_COMMENT = "\n\n% Fim de itemize\n\n"

ITEMIZE_TEMPLATE = """
\\begin{itemize}
$ITEMS
\end{itemize}
"""

parser = argparse.ArgumentParser(description='Topic itemizes generator')
parser.add_argument('--models', nargs='+', help='list of model files', required=True)
args = parser.parse_args()


def get_percentage(number):
    return round(number * 100, 2)

def get_topic_string(topic):
    topic_words = list(map(lambda x: f'\({get_percentage(x[0])}\%\) \\textbf{{{x[1]}}}', topic[:MAX_WORDS_PER_TOPIC]))
    return " + ".join(topic_words).replace(".", ",")

def get_topic_item(idx, topic):
    return f'\item \(T_{{{idx}}}\): {get_topic_string(topic)} ... -- '


def get_model_itemize(model_path):
    model = joblib.load(model_path)

    model_itemize = ITEMIZE_TEMPLATE
    
    lines = []
    for idx, topic in enumerate(model['topics_with_word_probs']):
        lines.append(get_topic_item(idx, topic))
    
    table_items = '\n'.join(lines)
    model_itemize = model_itemize.replace("$ITEMS", table_items)

    return model_itemize


itemizes = ""

for model_path in args.models:
    model_itemize = get_model_itemize(model_path)
    itemizes += START_COMMENT + model_itemize + END_COMMENT

print(itemizes)
