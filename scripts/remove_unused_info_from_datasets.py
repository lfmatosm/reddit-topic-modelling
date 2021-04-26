import json
import argparse
import os


def get_base_path(dataset_path):
    return os.path.sep.join(dataset_path.split(os.path.sep)[:-1])


def get_dataset_name(dataset_path):
    return dataset_path.split(os.path.sep)[-1]


def get_updated_name(dataset_path):
    path = get_base_path(dataset_path)
    name = get_dataset_name(dataset_path)
    return f'{path}/_{name}'


parser = argparse.ArgumentParser(description='Removes unused data from datasets, leaving just submission body content')
parser.add_argument('--datasets', nargs='+', help='list of datasets', required=True)
args = parser.parse_args()

datasets = args.datasets

for dataset in datasets:
    print(f'Processing "{dataset}"...')
    data = json.load(open(dataset, "r"))
    filtered_data = list(map(lambda document: {
        "clicked": document["clicked"],
        "created_utc": document["created_utc"],
        "date": document["date"],
        "distinguished": document["distinguished"],
        "edited": document["edited"],
        "id": document["id"],
        "is_original_content": document["is_original_content"],
        "is_text_only": document["is_text_only"],
        "link_flair_template_id": document["link_flair_template_id"],
        "link_flair_text": document["link_flair_text"],
        "locked": document["locked"],
        "name": document["name"],
        "num_comments": document["num_comments"],
        "over_18": document["over_18"],
        "permalink": document["permalink"],
        "score": document["score"],
        "body": document["body"],
        "spoiler": document["spoiler"],
        "stickied": document["stickied"],
        "subreddit_id": document["subreddit_id"],
        "subreddit_name": document["subreddit_name"],
        "title": document["title"],
        "upvote_ratio": document["upvote_ratio"],
        "url": document["url"],
    }, data))
    updated_name = get_updated_name(dataset)
    json.dump(filtered_data, open(updated_name, "w"))

print("Finished")
