from pymongo import MongoClient
import os

def get_db():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client["open_gait"]
    return db
