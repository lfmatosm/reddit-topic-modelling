import pandas as pd
from decimal import Decimal
import os

OUTPUT_PATH = "results/csv/lda/topics/"


def create_dataframe_from_topics(topic_word_dist, idx_to_word):
    pairs = get_topic_word_probability_pairs(topic_word_dist, idx_to_word)

    topics_dicts = list(map(lambda pair: {
        "palavras": " ".join(list(pair.keys())[::-1][0:10])
    }, pairs))
    
    return pd.DataFrame.from_records(topics_dicts)


def get_top_documents_for_each_topic(topics_df, original_df, doc_topic_distribution, threshold, topn_docs):
    df = get_document_topic_dataframe(doc_topic_distribution)
    
    grouped_df = df[df['probability'] >= threshold].groupby('topic')

    topic_files_df = pd.DataFrame({
        "topic": [],
        "topic file": [],
        "documents file": []
    })

    for topic in df['topic'].unique():
        group = grouped_df.get_group(topic)

        ordered_group = group.sort_values(by="probability", ascending=False).head(topn_docs)

        topic_documents = original_df[original_df.index.isin(ordered_group['document'])].copy()

        topic_documents["probability"] = group[group['document'].isin(ordered_group['document'])]["probability"]

        output_folder = OUTPUT_PATH + f'Topic_{topic}/'

        os.makedirs(os.path.dirname(output_folder), exist_ok=True)

        topic_words_path = output_folder + "topic_words.csv"

        topics_df[topics_df.index == topic].to_csv(topic_words_path)

        topic_documents_path = output_folder + "topic_docs.csv"

        topic_documents.sort_values(by="probability", ascending=False).to_csv(topic_documents_path)

        topic_files_df = topic_files_df.append({
            "topic": topic,
            "topic file": topic_words_path,
            "documents file": topic_documents_path
        }, ignore_index=True)
        
    return topic_files_df.sort_values(by="topic", ascending=True)


def get_document_topic_dataframe(doc_topic_distribution):
    df = pd.DataFrame({
        "topic": [],
        "document": [],
        "probability": []
    })

    for n in range(doc_topic_distribution.shape[0]):
        topic_most_pr = doc_topic_distribution[n].argmax()

        df = df.append({
            "topic": topic_most_pr,
            "document": n,
            "probability": doc_topic_distribution[n][topic_most_pr]
        }, ignore_index=True)

    return df


def get_topic_word_probability_pairs(topic_word_dist, idx_to_word):
    pairs = []

    for _, topic in enumerate(topic_word_dist):
        sorted_index = topic.argsort()[-20:]
        sorted_topic = topic[sorted_index]
        sorted_words = idx_to_word[sorted_index]

        topic_pairs = list(zip(sorted_words, sorted_topic))

        topic_word_dict = {}

        for pair in topic_pairs:
            topic_word_dict[pair[0]] = pair[1]

        pairs.append(topic_word_dict)
    
    return pairs
