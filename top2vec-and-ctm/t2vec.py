from top2vec import Top2Vec
import json

documents = [data["body"] for data in json.load(open("top2vec-and-ctm/reddit_gatherer.pt_submissions[original_dataset][2008_2020].json", "r"))]
print(documents[0])

model = Top2Vec(documents, workers=3)

n_topics = model.get_num_topics()
print(f'No of topics: {n_topics}')
topic_words, word_scores, topic_nums = model.get_topics(n_topics)
print(topic_words[0][0:10])

words, word_scores = model.similar_words(keywords=["suicidio"], keywords_neg=[], num_words=10)
for word, score in zip(words, word_scores):
    print(f"{word} {score}")
