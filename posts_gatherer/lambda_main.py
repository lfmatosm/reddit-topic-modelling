import sys
import os
import praw
import math
from datetime import datetime
from db.dynamo import get_last_searched_date, save_last_searched_date
from parsers.reddit_parser import get_comment_data, get_submission_data, get_subreddit_data, get_comments
from services.reddit_service import insert_comment, insert_submission, insert_subreddit
from integrations.pushshift import get_submissions_with_keywords_for_interval
from utils.time_interval import get_timestamp_interval_for_starting_date


DEFAULT_COLLECTIONS = {
    'SUBMISSIONS': 'submissions',
    'COMMENTS': 'comments',
    'SUBREDDITS': 'subreddits',
}

DATE_FORMAT = '%Y-%m-%d'

params = {
    'subreddits': os.getenv('SUBREDDITS'),
    'keywords': os.getenv('SEARCH_KEYWORDS'),
    'start': os.getenv('START_DATE'),
    'end': os.getenv('END_DATE'),
    'saveComments': bool(os.getenv('SAVE_COMMENTS')),
    'saveSubreddits': bool(os.getenv('SAVE_SUBREDDITS')),
    'submissionsCollection': DEFAULT_COLLECTIONS['SUBMISSIONS'],
    'commentsCollection': DEFAULT_COLLECTIONS['COMMENTS'],
    'subredditsCollection': DEFAULT_COLLECTIONS['SUBREDDITS'],
    'daysPerInterval': int(os.getenv('DAYS_PER_INTERVAL')),
}
print(f'Running on AWS ENV with params {params}')

start_date = get_last_searched_date()

print(f'Current start date: {start_date}')

max_end_date = datetime.strptime(params['end'], DATE_FORMAT)
days = params['daysPerInterval']
interval = get_timestamp_interval_for_starting_date(start_date, max_end_date, days)

print(f'Starting search within {datetime.fromtimestamp(interval[0])} - {datetime.fromtimestamp(interval[1])} date range')

subreddit_submissions_map = {}

for subreddit in params['subreddits']:
    submission_ids = set()

    print(f'Searching inside "{subreddit}" subreddit...')

    for keyword in params['keywords']:
        print(f'Searching for "{keyword}" keyword...')

        new_submission_ids = get_submissions_with_keywords_for_interval(keyword, subreddit, interval)

        if (len(new_submission_ids) == 0): continue

        submission_ids = submission_ids.union(new_submission_ids)

    subreddit_submissions_map[subreddit] = list(submission_ids)


all_ids = [sub_id for id_list in list(subreddit_submissions_map.values()) for sub_id in id_list]
total_submissions = len(all_ids)
print(f'{total_submissions} submissions found with the given keywords and within the given date range')

print(f'Starting gathering...')

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'), 
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    password=os.getenv('REDDIT_PASSWORD'), 
    user_agent=os.getenv('REDDIT_USERAGENT'),
    username=os.getenv('REDDIT_USERNAME')
)

subreddits = list(subreddit_submissions_map.keys())
for subreddit_id in subreddits:
    subreddit = reddit.subreddit(subreddit_id)

    if params['saveSubreddits'] == True:
        subreddit_data = get_subreddit_data(subreddit)
        insert_subreddit(
            subreddit_data, 
            params['subredditsCollection']
        )

    submissions = subreddit_submissions_map[subreddit_id]
    no_of_submissions_in_subreddit = len(submissions)
    print(f'Gathering {no_of_submissions_in_subreddit} posts on "{subreddit.display_name}" subreddit')

    for submission_id in submissions:
        submission = reddit.submission(submission_id)

        submission_data = get_submission_data(submission)
        if submission_data != None:
            insert_submission(
                submission_data,  
                params['submissionsCollection']
            )

        if params['saveComments'] == True:
            comments = get_comments(submission)
        
            for comment in comments:
                insert_comment(
                    comment, 
                    params['commentsCollection']
                )

print(f'Posts gathered and saved')

last_searched_date = datetime.fromtimestamp(interval[1])
save_last_searched_date(last_searched_date)

print(f'Last searched date saved: {last_searched_date}')

