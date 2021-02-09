import json
import requests
from datetime import datetime


PUSHSHIFT_URL = "https://api.pushshift.io/reddit/search/submission/"


def get_submissions_with_keywords_for_interval(keyword, subreddit, interval, size = 500):
    """Search for a keyword inside a subreddit within a time interval
    and returns the respective submission ids found. Pushshift API is used for searching.

    Parameters:

    keyword (str): keyword to search

    subreddit (str): subreddit title

    interval (tuple): interval object (tuple) representing starting timestamp and ending timestamp

    size (int) - optional: page size requested to the Pushshift API.

    Returns:

    list: a list of submission ids
    """
    request_url = f'{PUSHSHIFT_URL}?q={keyword}&subreddit={subreddit}&after={interval[0]}&before={interval[1]}&size={size}'

    response = requests.get(request_url)

    if (response.content == None):
        return []

    content = json.loads(response.content)

    return list(map(lambda submission: submission["id"], content["data"]))