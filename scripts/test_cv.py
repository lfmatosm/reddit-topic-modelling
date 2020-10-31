from sklearn.feature_extraction.text import CountVectorizer
import json

documents = json.load(open("data/processed/reddit_gatherer.pt_submissions[original_dataset][2008_2020]_[original_dataset_without_duplicates][processed].json", "r"))
joined_documents = [ " ".join(document["body"]) for document in documents ]

vectorizer = CountVectorizer(min_df=0.05, max_df=0.6)

vectorized_documents = vectorizer.fit_transform(joined_documents)
print(vectorized_documents)
print(len(vectorizer.vocabulary_))
print(len(vectorizer.stop_words_))

print((vectorizer.stop_words_))

print([[word for word in document.split() if word not in vectorizer.stop_words_] for document in joined_documents])