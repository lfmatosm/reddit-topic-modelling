import os
import joblib
import argparse


MAX_MINIPAGES_PER_FIGURE = 8
START_COMMENT = "% Inicio de tabela\n\n"
END_COMMENT = "\n\n% Fim de tabela\n\n"

FIGURE_TEMPLATE = """
\\begin{figure}[!htb]
\centering
$MINIPAGES
\end{figure}
"""

MINIPAGE_TEMPLATE = """\\begin{minipage}{.48\\textwidth}
\includegraphics[scale=0.6]{$IMG_PATH}
\caption{$CAPTION}
\label{$LABEL}
\end{minipage}"""

parser = argparse.ArgumentParser(description='Topic figures generator')
parser.add_argument('--model', type=str, help='model file', required=True)
parser.add_argument('--lang', type=str, help='model file', required=True)
args = parser.parse_args()


def get_model_name(model_path):
    return model_path.split(os.path.sep)[-1]


def get_model_type(model_path):
    model_name = get_model_name(model_path)
    return model_name.split('_')[0].lower()


def get_topic_minipage(idx, topic, model_lang, model_type):
    img_path = f'imagens/modelos/topicos/{model_type}/{model_lang}/{model_type}_topic={idx+1}.pdf'
    caption = f'\(T_{{{idx}}}\) do modelo {model_type.upper()}'
    label = f'fig:{model_type}{model_lang.capitalize()}Tp{idx}'
    return MINIPAGE_TEMPLATE.replace("$IMG_PATH", img_path).replace("$CAPTION", caption).replace("$LABEL", label)


def get_model_topic_images(model_path, model_lang):
    model = joblib.load(model_path)
    model_type = get_model_type(model_path)

    minipages = []
    for idx, topic in enumerate(model["topics"]):
        minipages.append(get_topic_minipage(idx, topic, model_lang, model_type))
    
    minipages_pages = [minipages[i:i+MAX_MINIPAGES_PER_FIGURE] for i in range(0, len(minipages), MAX_MINIPAGES_PER_FIGURE)]

    figures = []
    for page in minipages_pages:
        minipages_temp = '\hfill\n'.join(page)
        figures.append(FIGURE_TEMPLATE.replace("$MINIPAGES", minipages_temp))
    
    return END_COMMENT.join(figures)


code = ""

model_figures = get_model_topic_images(args.model, args.lang)
code = model_figures

print(code)
