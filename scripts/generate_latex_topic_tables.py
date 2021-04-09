import os
import joblib
import argparse

MAX_WORDS_PER_TOPIC = 10
START_COMMENT = "% Inicio de tabela\n\n"
END_COMMENT = "\n\n% Fim de tabela\n\n"

TABLE_TEMPLATE = """
\\begin{table}[ht]
\centering
\caption{$TABLE_CAPTION}
\label{$TABLE_LABEL}
\\begin{tabular}{| c | c |}
\hline
$TABLE_LINES
\end{tabular}
\end{table}
"""

parser = argparse.ArgumentParser(description='Topic tables generator')
parser.add_argument('--models', nargs='+', help='list of model files', required=True)
args = parser.parse_args()


def get_model_name(model_path):
    return model_path.split(os.path.sep)[-1]


def get_model_type(model_path):
    model_name = get_model_name(model_path)
    return model_name.split('_')[0].upper()


def get_topic_line(idx, topic):
    topic_str = ' '.join(topic[:MAX_WORDS_PER_TOPIC])
    return f'\(T_{{{idx}}}\) & {topic_str} \\\\ \hline'


def get_model_table(model_path):
    model = joblib.load(model_path)

    model_table = TABLE_TEMPLATE
    
    model_type = get_model_type(model_path)
    lines = []
    for idx, topic in enumerate(model['topics']):
        lines.append(get_topic_line(idx, topic))
    
    table_lines = '\n'.join(lines)
    table_label = f'tab:modelo{model_type.lower().capitalize()}'
    table_caption = f'Tópicos do modelo {model_type} com \(k={len(model["topics"])}\).'
    model_table = model_table.replace("$TABLE_LINES", table_lines)
    model_table = model_table.replace("$TABLE_LABEL", table_label)
    model_table = model_table.replace("$TABLE_CAPTION", table_caption)

    return model_table


tables = ""

for model_path in args.models:
    model_table = get_model_table(model_path)
    tables += START_COMMENT + model_table + END_COMMENT

print(tables)