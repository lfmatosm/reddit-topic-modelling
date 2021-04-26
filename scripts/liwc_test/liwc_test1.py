from collections import Counter
import liwc
import re

parse, category_names = liwc.load_token_parser('LIWC2007_Portugues_win.dic')


def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

gettysburg = '''Você está vivendo um episódio de depressão. Eu sei porque já estive aí e pode ser que volte em alguma hora. Bem, quero aproveitar este momento em que me encontro aqui do lado de fora para te escrever esta carta com base no eu que já esteve aí e neste outro que um dia pode voltar.

Não tenho a pretensão de te ajudar. Quero apenas te fazer companhia por alguns momentos, como um desconhecido que se senta do teu lado num banco público, reconhece a tua solidão e puxa uma conversa, tentando respeitar sua aflição e tomando cuidado para não perturbar a segurança precária do teu isolamento.'''.lower()
gettysburg_tokens = tokenize(gettysburg)

gettysburg_counts = Counter(category for token in gettysburg_tokens for category in parse(token))
print(gettysburg_counts)
