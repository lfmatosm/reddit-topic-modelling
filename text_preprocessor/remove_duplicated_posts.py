import argparse, json, os, pandas as pd


def remove_bots_posts(dataset):
    bots = ["AutoModerator", "RemindMeBot", "WikiTextBot", "youtubefactsbot", "RedditNiobioBot", "NemLiNemLereiBot"]

    return list(filter(lambda data: (not hasattr(data, 'author')) or (data['author'] == None) or (data['author'] != None and data['author']['name'] not in bots), dataset))


parser = argparse.ArgumentParser(description='Removes duplicated, removed and empty posts.')

parser.add_argument('--dataset', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--datasetName', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--outputPath', type=str, help='path to put the resulting split datasets', required=True)

args = parser.parse_args()

original_dataset = json.load(open(args.dataset, 'r'))

print("Original row count: ", len(original_dataset))

original_dataset = remove_bots_posts(original_dataset)

print("Row count after bots' posts removal: ", len(original_dataset))

original_data_frame = pd.DataFrame.from_dict(original_dataset)

df_without_duplicates = original_data_frame.drop_duplicates(subset=['body'], keep='first')

print("Row count after duplicates removal: ", len(df_without_duplicates))

df_deleted_posts_removed = df_without_duplicates[df_without_duplicates.body != "[deleted]"]

df_removed_posts_removed = df_deleted_posts_removed[df_deleted_posts_removed.body != "[removed]"]

print("Row count after deleted/removed posts removal: ", len(df_removed_posts_removed))

df_empty_posts_removed = df_removed_posts_removed[df_removed_posts_removed.body != ""]

print("Row count after empty posts removal: ", len(df_empty_posts_removed))

original_dataset = df_empty_posts_removed.to_dict(orient='records')

path = os.path.join(args.outputPath, f'{args.datasetName}_[without_duplicates].json')
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(original_dataset, open(path, 'w'))

print(f'Dataset saved to {path} path.')
