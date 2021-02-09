from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import os
import praw
import math
from datetime import datetime
from parsers.reddit_parser import get_comment_data, get_submission_data, get_subreddit_data, get_comments
from services.reddit_service import insert_comment, insert_submission, insert_subreddit
from integrations.pushshift import get_submissions_with_keywords_for_interval
from utils.progress_bar import update_progress_bar
from utils.time_interval import get_timestamps_interval


DEFAULT_COLLECTIONS = {
    'SUBMISSIONS': 'submissions',
    'COMMENTS': 'comments',
    'SUBREDDITS': 'subreddits',
}

DATE_FORMAT = '%Y-%m-%d'


def get_all_submissions_from_intervals(keyword, subreddit, intervals, size = 500):
    """Search for a keyword inside a subreddit within time intervals
    and returns the respective submission ids found.

    Parameters:

    keyword (str): keyword to search

    subreddit (str): subreddit title

    intervals (list of tuple): list of interval objects (tuples) representing starting timestamp and ending timestamp

    size (int) - optional: page size requested to the Pushshift API.

    Returns:

    list: a list of submission ids
    """
    ids = []

    for interval in intervals:
        start_date = datetime.fromtimestamp(interval[0])
        end_date = datetime.fromtimestamp(interval[1])
        print(f'Searching keyword within range ({start_date}, {end_date})...')

        submission_ids = get_submissions_with_keywords_for_interval(keyword, subreddit, interval)

        new_ids_without_duplicates = set(submission_ids) - set(ids)

        ids = ids + list(new_ids_without_duplicates)

    return ids


parser = argparse.ArgumentParser(description='Gather Reddit submission data and sends to cloud database.')

parser.add_argument('--subreddits', nargs='+', help='subreddits to gather', required=True)
parser.add_argument('--keywords', nargs='+', help='keywords to search for on the subreddit', required=True)
parser.add_argument('--searchResultsPath', type=str, help='path to save the list of submission ids returned by the search', required=False, default='search/results.txt')
parser.add_argument('--start', type=str, help='gather posts written after this date', required=True)
parser.add_argument('--end', type=str, help='gather posts written before this date', required=True)
parser.add_argument('--saveComments', type=int, help='wheter should save submission comments', required=False, default=False)
parser.add_argument('--saveSubreddits', type=int, help='wheter should save submission subreddit', required=False, default=False)
parser.add_argument('--submissionsCollection', type=str, help='MongoDB collection to save submissions', required=False, default=DEFAULT_COLLECTIONS['SUBMISSIONS'])
parser.add_argument('--commentsCollection', type=str, help='MongoDB collection to save comments', required=False, default=DEFAULT_COLLECTIONS['COMMENTS'])
parser.add_argument('--subredditsCollection', type=str, help='MongoDB collection to save subreddits', required=False, default=DEFAULT_COLLECTIONS['SUBREDDITS'])
parser.add_argument('--daysPerInterval', type=int, help='no. of days per search interval', required=False)

args = parser.parse_args()
params = {
    'subreddits': args.subreddits,
    'keywords': args.keywords,
    'searchResultsPath': args.searchResultsPath,
    'start': args.start,
    'end': args.end,
    'saveComments': bool(args.saveComments),
    'saveSubreddits': bool(args.saveSubreddits),
    'submissionsCollection': args.submissionsCollection,
    'commentsCollection': args.commentsCollection,
    'subredditsCollection': args.subredditsCollection,
    'daysPerInterval': args.daysPerInterval,
}


startDate = datetime.strptime(params['start'], DATE_FORMAT)
endDate = datetime.strptime(params['end'], DATE_FORMAT)
days = params['daysPerInterval']
timestampsInterval = list(get_timestamps_interval(startDate, endDate, days_per_interval=days) \
    if (days != None) else get_timestamps_interval(startDate, endDate))


print(f'Starting search...')

subreddit_submissions_map = {}

for subreddit in params['subreddits']:
    submission_ids = []

    print(f'Searching inside "{subreddit}" subreddit...')

    for keyword in params['keywords']:
        print(f'Searching for "{keyword}" keyword...')

        new_submission_ids = get_all_submissions_from_intervals(keyword, subreddit, timestampsInterval)

        if (len(new_submission_ids) == 0): continue

        original_ids_set = set(submission_ids)
        new_ids_set = set(new_submission_ids)
        new_ids_without_duplicates = new_ids_set - original_ids_set

        submission_ids = submission_ids + list(new_ids_without_duplicates)

    subreddit_submissions_map[subreddit] = submission_ids


all_ids = [sub_id for id_list in list(subreddit_submissions_map.values()) for sub_id in id_list]
total_submissions = len(all_ids)
print(f'{total_submissions} submissions found with the given keywords ({", ".join(params["keywords"])}) and within the date range ({startDate.date()}, {endDate.date()})')


print(f'Start gathering...')

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'), 
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    password=os.getenv('REDDIT_PASSWORD'), 
    user_agent=os.getenv('REDDIT_USERAGENT'),
    username=os.getenv('REDDIT_USERNAME')
)

subreddits = list(subreddit_submissions_map.keys())
for k in range(len(subreddits)):
    subreddit = reddit.subreddit(subreddits[k])
    print("Subreddit name: " + subreddit.display_name)

    if params['saveSubreddits'] == True:
        subreddit_data = get_subreddit_data(subreddit)
        insert_subreddit(
            subreddit_data, 
            params['subredditsCollection']
        )

    submissions = subreddit_submissions_map[subreddits[k]]
    no_of_submissions_in_subreddit = len(submissions)
    print(f'Gathering: {no_of_submissions_in_subreddit} out of {total_submissions} submissions')

    for i in range(no_of_submissions_in_subreddit):
        update_progress_bar(i, no_of_submissions_in_subreddit)

        submission = reddit.submission(submissions[i])

        submission_data = get_submission_data(submission)
        if submission_data != None:
            insert_submission(
                submission_data, 
                params['submissionsCollection']
            )


        if params['saveComments'] == True:
            comments = get_comments(submission)

            total_comments = len(comments)
        
            for j in range(total_comments):
                insert_comment(
                    comments[j], 
                    params['commentsCollection']
                )
    
    update_progress_bar(no_of_submissions_in_subreddit, no_of_submissions_in_subreddit)
        


print("\nFinished gathering.")
