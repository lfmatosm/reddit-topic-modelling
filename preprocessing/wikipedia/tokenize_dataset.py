from gensim.utils import simple_preprocess
import argparse

parser = argparse.ArgumentParser(description='Wikipedia dataset tokenizer')
parser.add_argument('--dataset', type=str, help='dataset file', required=True)
args = parser.parse_args()


def load_dataset(dataset):
    with open(dataset, "r") as file:
        for line in file:
            yield line.strip("\n")


def save_dataset(dataset, tokenized_data):
    with open(dataset.strip(".txt") + "_tokenized" + ".txt", "w") as file:
        for line in tokenized_data:
            file.write(" ".join(line).strip("\n"))


texts = load_dataset(args.dataset)
tokenize = lambda texts: [(yield simple_preprocess(text, deacc=True, min_len=1)) for text in texts]
tokenized_data = tokenize(texts)
save_dataset(args.dataset, tokenized_data)
