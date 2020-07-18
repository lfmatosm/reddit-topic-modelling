import json
import requests
import numpy as np
from datetime import datetime


PUSHSHIFT_URL = "https://api.pushshift.io/reddit/search/submission/"


def get_submissions_with_keyword(keyword, subreddit, intervals, size = 500):
    """Search for a keyword inside a subreddit within time intervals
    and returns the respective submission ids found. Pushshift API is used for searching.

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

        request_url = f'{PUSHSHIFT_URL}?q={keyword}&subreddit={subreddit}&after={interval[0]}&before={interval[1]}&size={size}'

        response = requests.get(request_url)

        if (response.content == None):
            continue

        content = json.loads(response.content)

        submission_ids = list(map(lambda submission: submission["id"], content["data"]))

        unique_ids = list(np.unique(submission_ids))

        new_ids_without_duplicates = set(unique_ids) - set(ids)

        ids = ids + list(new_ids_without_duplicates)

    return ids
