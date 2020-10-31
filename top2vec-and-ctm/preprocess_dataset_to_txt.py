import json

# documents = [data["body"] for data in json.load(open("top2vec-and-ctm/reddit_gatherer.pt_submissions[original_dataset][2008_2020].json", "r"))]
# documents = [" ".join(data["body"]) for data in json.load(open("data/processed/reddit_gatherer.pt_submissions[original_dataset][2008_2020][nouns].json", "r"))]
documents = [" ".join(data) for data in json.load(open("datasets_for_training/training_dataset.json", "r"))]


with open("dataset_pt_nouns_2008_2020.txt", "w") as file:
    for document in documents:
        doc_txt = document.replace("\n", "")
        file.write(doc_txt + "\n")
