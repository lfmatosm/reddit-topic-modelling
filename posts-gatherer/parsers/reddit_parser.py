from datetime import datetime


def get_author_data(author):
    """Creates a author object from a PRAW Redditor instance

    Parameters:

    author (praw.models.Redditor): PRAW Redditor instance

    Returns:

    dict: object with information about a Reddit user, like his name or id
    """
    try:
        if (author == None): 
            return None

        return {
            "name": author.name if hasattr(author, "name") else None,
            "id": author.id if hasattr(author, "id") else None,
            "comment_karma": author.comment_karma if hasattr(author, "comment_karma") else None,
            "created_utc": author.created_utc if hasattr(author, "created_utc") else None,
            "is_suspended": author.is_suspended if hasattr(author, "is_suspended") else None,
            "is_mod": author.is_mod if hasattr(author, "is_mod") else None,
            "is_employee": author.is_employee if hasattr(author, "is_employee") else None,
            "has_verified_email": author.has_verified_email if hasattr(author, "has_verified_email") else None
        }
    except:
        return None


def get_comment_data(raw_comment):
    """Creates a comment object from a PRAW Comment instance

    Parameters:

    raw_comment (praw.models.Comment): PRAW Comment instance

    Returns:

    dict: object with information about a comment, like body, author, permalink or score
    """
    if (raw_comment.body == "") or (raw_comment.body == "[deleted]"):
        return None

    author = get_author_data(raw_comment.author)
    date = datetime.fromtimestamp(raw_comment.created_utc) if hasattr(raw_comment, 'created_utc') else None
    has_submission = hasattr(raw_comment, 'submission')
    has_subreddit = hasattr(raw_comment, 'subreddit')

    return {
        "author": author,
        "body": raw_comment.body if hasattr(raw_comment, 'body') else None,
        "created_utc": raw_comment.created_utc if hasattr(raw_comment, 'created_utc') else None,
        "date": date.strftime('%Y-%m-%d %H:%M:%S') if date != None else None,
        "distinguished": raw_comment.distinguished if hasattr(raw_comment, 'distinguished') else None,
        "edited": raw_comment.edited if hasattr(raw_comment, 'edited') else None,
        "id": raw_comment.id if hasattr(raw_comment, 'id') else None,
        "is_submitter": raw_comment.is_submitter if hasattr(raw_comment, 'is_submitter') else None,
        "link_id": raw_comment.link_id if hasattr(raw_comment, 'link_id') else None,
        "parent_id": raw_comment.parent_id if hasattr(raw_comment, 'parent_id') else None,
        "permalink": raw_comment.permalink if hasattr(raw_comment, 'permalink') else None,
        "score": raw_comment.score if hasattr(raw_comment, 'score') else None,
        "stickied": raw_comment.stickied if hasattr(raw_comment, 'stickied') else None,
        "submission_id": raw_comment.submission.id if has_submission and hasattr(raw_comment.submission, 'id') else None,
        "submission_name": raw_comment.submission.name if has_submission and hasattr(raw_comment.submission, 'name') else None,
        "submission_url": raw_comment.submission.url if has_submission and hasattr(raw_comment.submission, 'url') else None,
        "subreddit_id": raw_comment.subreddit.id if has_subreddit and hasattr(raw_comment.subreddit, 'id') else None,
        "subreddit_name": raw_comment.subreddit.name if has_subreddit and hasattr(raw_comment.subreddit, 'name') else None
    }


def get_submission_data(raw_submission):
    """Creates a submission object from a PRAW Submission instance

    Parameters:

    raw_submission (praw.models.Submission): PRAW Submission instance

    Returns:

    dict: object with information about a submission, like body, author or URL
    """
    if raw_submission.selftext == "[deleted]":
        return None

    author = get_author_data(raw_submission.author)
    date = datetime.fromtimestamp(raw_submission.created_utc) if hasattr(raw_submission, 'created_utc') else None
    has_subreddit = hasattr(raw_submission, 'subreddit')

    return {
        "author": author,
        "clicked": raw_submission.clicked if hasattr(raw_submission, 'clicked') else None,
        "created_utc": raw_submission.created_utc if hasattr(raw_submission, 'created_utc') else None,
        "date": date.strftime('%Y-%m-%d %H:%M:%S') if date != None else None,
        "distinguished": raw_submission.distinguished if hasattr(raw_submission, 'distinguished') else None,
        "edited": raw_submission.edited if hasattr(raw_submission, 'edited') else None,
        "id": raw_submission.id if hasattr(raw_submission, 'id') else None,
        "is_original_content": raw_submission.is_original_content if hasattr(raw_submission, 'is_original_content') else None,
        "is_text_only": raw_submission.is_self if hasattr(raw_submission, 'is_self') else None,
        "link_flair_template_id": raw_submission.link_flair_template_id if hasattr(raw_submission, 'link_flair_template_id') else None,
        "link_flair_text": raw_submission.link_flair_text if hasattr(raw_submission, 'link_flair_text') else None,
        "locked": raw_submission.locked if hasattr(raw_submission, 'locked') else None,
        "name": raw_submission.name if hasattr(raw_submission, 'name') else None,
        "num_comments": raw_submission.num_comments if hasattr(raw_submission, 'num_comments') else None,
        "over_18": raw_submission.over_18 if hasattr(raw_submission, 'over_18') else None,
        "permalink": raw_submission.permalink if hasattr(raw_submission, 'permalink') else None,
        "score": raw_submission.score if hasattr(raw_submission, 'score') else None,
        "body": raw_submission.selftext if hasattr(raw_submission, 'selftext') else None,
        "spoiler": raw_submission.spoiler if hasattr(raw_submission, 'spoiler') else None,
        "stickied": raw_submission.stickied if hasattr(raw_submission, 'stickied') else None,
        "subreddit_id": raw_submission.subreddit.id if has_subreddit and hasattr(raw_submission.subreddit, 'id') else None,
        "subreddit_name": raw_submission.subreddit.name if has_subreddit and hasattr(raw_submission.subreddit, 'name') else None,
        "title": raw_submission.title if hasattr(raw_submission, 'title') else None,
        "upvote_ratio": raw_submission.upvote_ratio if hasattr(raw_submission, 'upvote_ratio') else None,
        "url": raw_submission.url if hasattr(raw_submission, 'url') else None
    }
    

def get_subreddit_data(raw_subreddit):
    """Creates a subreddit object from a PRAW Subreddit instance

    Parameters:

    raw_subreddit (praw.models.Subreddit): PRAW Subreddit instance

    Returns:

    dict: object with information about a subreddit
    """
    date = datetime.fromtimestamp(raw_subreddit.created_utc) if hasattr(raw_subreddit, 'created_utc') else None

    return {
        "date": date.strftime('%Y-%m-%d %H:%M:%S') if date != None else None,
        "can_assign_link_flair": raw_subreddit.can_assign_link_flair if hasattr(raw_subreddit, 'can_assign_link_flair') else None,
        "can_assign_user_flair": raw_subreddit.can_assign_user_flair if hasattr(raw_subreddit, 'can_assign_user_flair') else None,
        "created_utc": raw_subreddit.created_utc if hasattr(raw_subreddit, 'created_utc') else None,
        "description": raw_subreddit.description if hasattr(raw_subreddit, 'description') else None,
        "description_html": raw_subreddit.description_html if hasattr(raw_subreddit, 'description_html') else None,
        "display_name": raw_subreddit.display_name if hasattr(raw_subreddit, 'display_name') else None,
        "id": raw_subreddit.id if hasattr(raw_subreddit, 'id') else None,
        "name": raw_subreddit.name if hasattr(raw_subreddit, 'name') else None,
        "over18": raw_subreddit.over18 if hasattr(raw_subreddit, 'over18') else None,
        "public_description": raw_subreddit.public_description if hasattr(raw_subreddit, 'public_description') else None,
        "spoilers_enabled": raw_subreddit.spoilers_enabled if hasattr(raw_subreddit, 'spoilers_enabled') else None,
        "subscribers": raw_subreddit.subscribers if hasattr(raw_subreddit, 'subscribers') else None
    }
    