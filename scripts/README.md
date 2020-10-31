docker run --name word_embd_db -d -p 27017:27017 mongo

python scripts/create_embeddings_database.py --embeddings etm-trainer/skip_s300.txt --port 27017