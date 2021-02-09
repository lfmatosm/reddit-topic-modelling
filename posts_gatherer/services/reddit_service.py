from db.mongo import mongo_db


def insert_subreddit(subreddit, collection):
    """Inserts subreddit object on database.

    Parameters:
    
    subreddit (dict): subreddit object

    collection (str): name of the collection where the object should be saved
    """
    mongo_db[collection].insert_one(subreddit)


def insert_submission(submission, collection):
    """Inserts submission object on database.

    Parameters:
    
    submission (dict): submission object

    collection (str): name of the collection where the object should be saved
    """
    mongo_db[collection].insert_one(submission)


def insert_comment(comment, collection):
    """Inserts comment object on database.

    Parameters:
    
    comment (dict): comment object

    collection (str): name of the collection where the object should be saved
    """
    mongo_db[collection].insert_one(comment)
