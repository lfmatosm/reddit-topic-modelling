from dotenv import load_dotenv
load_dotenv()

import os
from pymongo import MongoClient

client = MongoClient(os.getenv("MONGODB_URL"))
mongo_db = client.reddit_posts_gatherer
