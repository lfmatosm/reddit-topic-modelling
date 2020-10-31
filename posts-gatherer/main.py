from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import os
import praw
import math
from datetime import datetime
from db.database import db
from submission.gatherer import get_submissions_with_keyword
from parsers.reddit_parser import get_comment_data, get_submission_data, get_subreddit_data
from utils.utils import update_progress_bar


DATE_FORMAT = '%d-%m-%Y'


def insert_subreddit(subreddit, subreddit_no, total_subreddit, collection):
    """Inserts subreddit object on database.

    Parameters:
    
    subreddit (dict): subreddit object

    subreddit_no (int): subreddit number

    total_subreddit (int): total number of subreddits evaluated

    collection (str): name of the collection where the object should be saved
    """
    db[collection].insert_one(subreddit)


def insert_submission(submission, submission_no, total_submissions, collection):
    """Inserts submission object on database.

    Parameters:
    
    submission (dict): submission object

    submission_no (int): submission number

    total_submissions (int): total number of submissions evaluated

    collection (str): name of the collection where the object should be saved
    """
    db[collection].insert_one(submission)


def insert_comment(comment, comment_no, total_comments, submission_no, total_submissions, collection):
    """Inserts comment object on database.

    Parameters:
    
    comment (dict): comment object

    comment_no (int): comment number

    total_comments (int): total number of comments evaluated

    submission_no (int): submissions number

    total_submissions (int): total number of submissions evaluated

    collection (str): name of the collection where the object should be saved
    """
    db[collection].insert_one(comment)


def get_comments(submission):
    """Get all comments from submission, regardless of its place on the discussion hierarchy.

    Parameters:
    
    submission (praw.models.Submission): PRAW submission instance

    Returns:

    list of comment objects: list of non-empty comments
    """
    submission.comments.replace_more(limit=None)

    result = map(lambda raw_comment: get_comment_data(raw_comment),
                 submission.comments.list())

    return list(filter(lambda x: x != None, result))


def get_timestamps_interval(start_date, end_date, days_per_interval = 90):
    """Creates a timestamps interval list, where each element is a pair (startingTimestamp, endingTimestamp).
    The intervals are split by the desired quantity of days inside each one. 

    Parameters:
    
    start_date (datetime): initial date of interval

    end_date (datetime): final date of interval

    days_per_interval (int) - optional: no of days per timestamp interval

    Returns:

    list of tuples: list of (startingTimestamp, endingTimestamp) pairs
    """
    start_timestamp = math.floor(start_date.timestamp())
    end_timestamp = math.ceil(end_date.timestamp())
        
    ## 1 day = 86400
    period = 86400 * days_per_interval

    start_at = start_timestamp
    end_at = start_at + period
    yield (int(start_at), int(end_at))

    padding = 1
    while end_at + period <= end_timestamp:
        start_at = end_at + padding
        end_at = (start_at - padding) + period
        yield (int(start_at), int(end_at))
    
    start_at = end_at + padding
    end_at = end_timestamp
    yield (int(start_at), int(end_at))



parser = argparse.ArgumentParser(description='Gather Reddit submission data and sends to cloud database.')

parser.add_argument('--subreddits', nargs='+', help='subreddits to gather', required=True)
parser.add_argument('--keywords', nargs='+', help='keywords to search for on the subreddit', required=True)
parser.add_argument('--searchResultsPath', type=str, help='path to save the list of submission ids returned by the search', required=False, default='search/results.txt')
parser.add_argument('--start', type=str, help='gather posts written after this date', required=True)
parser.add_argument('--end', type=str, help='gather posts written before this date', required=True)
parser.add_argument('--submissionsCollection', type=str, help='MongoDB collection to save submissions', required=True)
# parser.add_argument('--commentsCollection', type=str, help='MongoDB collection to save comments', required=True)
# parser.add_argument('--subredditsCollection', type=str, help='MongoDB collection to save subreddits', required=True)
parser.add_argument('--daysPerInterval', type=int, help='no. of days per search interval', required=False)

args = parser.parse_args()

startDate = datetime.strptime(args.start, DATE_FORMAT)
endDate = datetime.strptime(args.end, DATE_FORMAT)
days = args.daysPerInterval
timestampsInterval = list(get_timestamps_interval(startDate, endDate, days_per_interval=days) \
    if (days != None) else get_timestamps_interval(startDate, endDate))


print(f'Starting search...')

subreddit_submissions_map = {}

for subreddit in args.subreddits:
    submission_ids = []

    print(f'Searching inside "{subreddit}" subreddit...')

    for keyword in args.keywords:
        print(f'Searching for "{keyword}" keyword...')

        new_submission_ids = get_submissions_with_keyword(keyword, subreddit, timestampsInterval)

        if (len(new_submission_ids) == 0): continue

        original_ids_set = set(submission_ids)
        new_ids_set = set(new_submission_ids)
        new_ids_without_duplicates = new_ids_set - original_ids_set

        submission_ids = submission_ids + list(new_ids_without_duplicates)

    subreddit_submissions_map[subreddit] = submission_ids


all_ids = [sub_id for id_list in list(subreddit_submissions_map.values()) for sub_id in id_list]
total_submissions = len(all_ids)
print(f'{total_submissions} submissions found with the given keywords ({", ".join(args.keywords)}) and within the date range ({startDate.date()}, {endDate.date()})')


if total_submissions > 0:
    print(f'Saving search results ids to file: {args.searchResultsPath}...')
    os.makedirs(os.path.dirname(args.searchResultsPath), exist_ok=True)
    with open(args.searchResultsPath, 'w') as file:
        file.write(' '.join(all_ids))
    print(f'Search results saved.')
else:
    print(f'No submissions found!')
    sys.exit(0)


print(f'Start gathering...')

reddit = praw.Reddit(client_id=os.getenv("CLIENT_ID"), client_secret=os.getenv("CLIENT_SECRET"),
                    password=os.getenv("PASSWORD"), user_agent=os.getenv("USERAGENT"),
                    username=os.getenv("REDDIT_USERNAME"))

subreddits = list(subreddit_submissions_map.keys())
for k in range(len(subreddits)):
    subreddit = reddit.subreddit(subreddits[k])
    print("Subreddit name: " + subreddit.display_name)

    # subreddit_data = get_subreddit_data(subreddit)
    # insert_subreddit(subreddit_data, k+1, len(subreddits), args.subredditsCollection)

    submissions = subreddit_submissions_map[subreddits[k]]
    no_of_submissions_in_subreddit = len(submissions)
    print(f'Gathering: {no_of_submissions_in_subreddit} out of {total_submissions} submissions')

    for i in range(no_of_submissions_in_subreddit):
        update_progress_bar(i, no_of_submissions_in_subreddit)

        submission = reddit.submission(submissions[i])

        submission_data = get_submission_data(submission)
        if submission_data != None:
            insert_submission(submission_data, i+1, no_of_submissions_in_subreddit, args.submissionsCollection)

        # comments = get_comments(submission)

        # total_comments = len(comments)
    
        # for j in range(total_comments):
        #     insert_comment(comments[j], j+1, total_comments, i+1, no_of_submissions_in_subreddit, args.commentsCollection)

    update_progress_bar(no_of_submissions_in_subreddit, no_of_submissions_in_subreddit)
        


print("\nFinished gathering.")
