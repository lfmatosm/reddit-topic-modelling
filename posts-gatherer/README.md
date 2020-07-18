# posts-gatherer
An application to gather Reddit comments data.

The gatherer works like this: you specify subreddits, time period and keywords to search for, and the script will gather comments, submission and subreddit data for you. Also, the script will save the gathered data to a MongoDB database of your choice, given a connection string, inside collections for each kind of data.

### Setup
On the project root folder, run:
```pipenv install```

Then activate the Pipenv environment with: ```pipenv shell```

On the ```src``` folder, create an ```.env``` with your environment variables. The file should have the following structure:

```CLIENT_ID=<your_client_id>
CLIENT_SECRET=<your_client_secret>
PASSWORD=<your_reddit_password>
USERAGENT=<your_reddit_app_user_agente>
REDDIT_USERNAME=<your_reddit_username>
MONGODB_URL=<your_mongodb_url_to_save_the_data>
```

### Running
You just need to execute ```python3 src/main.py``` script. Note that the ```.env``` should be on the same folder as the script.

The script needs some arguments to be passed on. These are the following:

* ```--subreddits``` - list of subreddits to search and gather data
* ```--keywords``` - list of keywords to search for on the specified subreddits
* ```--start``` - search for content created after this date. Format is ```DD-MM-YYYY```
* ```--end``` - search for content created before this date. Format is ```DD-MM-YYYY```
* ```--submissionsCollection``` - collection to save the gathered submissions' data
* ```--commentsCollection``` - collection to save the gathered comments' data
* ```--subredditsCollection``` - collection to save the gathered subreddits' data

Below, a command example, with all the arguments mentioned above:

```python3 src/main.py --subreddits brasil desabafos --keywords depressão suicídio "diagnóstico depressão" "tratamento depressão" depressivo --start 23-06-2005 --end 18-06-2020 --submissionsCollection pt_rd_submissions --commentsCollection pt_rd_comments --subredditsCollection pt_rd_subreddits```

### Docs

* [PRAW docs](https://praw.readthedocs.io/en/latest/index.html) - with examples and class specs.
* [Extracting comments from submissions](https://praw.readthedocs.io/en/latest/tutorials/comments.html) - specifically the subject of this script.
* [Connecting a Python application to MongoDB](https://www.mongodb.com/blog/post/getting-started-with-python-and-mongodb)
