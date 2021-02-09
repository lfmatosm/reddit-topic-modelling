import math
from datetime import datetime


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
        
    # 1 day = 86400
    period = 86400 * days_per_interval

    start_at = start_timestamp
    end_at = start_at + period
    yield (start_at, end_at)

    padding = 1
    while end_at + period <= end_timestamp:
        start_at = end_at + padding
        end_at = (start_at - padding) + period
        yield (start_at, end_at)
    
    start_at = end_at + padding
    end_at = end_timestamp
    yield (start_at, end_at)


def get_timestamp_interval_for_starting_date(start_date, max_end_date, days_per_interval = 90):
    start_timestamp = math.floor(start_date.timestamp())
    max_end_timestamp = math.ceil(max_end_date.timestamp())
        
    # 1 day = 86400
    period = 86400 * days_per_interval

    start_at = start_timestamp
    end_at = start_at + period
    end_at = end_at if end_at < max_end_timestamp else max_end_timestamp

    return (start_at, end_at)
