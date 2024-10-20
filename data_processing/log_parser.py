import json
from sentence_transformers import SentenceTransformer

def generate_embeddings(logs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [log['text'] for log in logs]
    embeddings = model.encode(texts)
    return embeddings

