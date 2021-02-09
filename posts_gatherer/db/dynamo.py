import os

def get_last_searched_date():
    return os.getenv('START_DATE')


def save_last_searched_date(last_searched_date):
    return None
