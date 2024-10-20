import json
from sentence_transformers import SentenceTransformer

# Load the dummy logs
def load_logs(file_path):
    with open(file_path, 'r') as f:
        logs = json.load(f)
    return logs

# Generate embeddings for the log texts
def generate_embeddings(logs):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # A small and fast model for embeddings
    texts = [log['text'] for log in logs]
    embeddings = model.encode(texts)
    return embeddings

if __name__ == "__main__":
    logs = load_logs('data/logs/dummy_logs.json')
    embeddings = generate_embeddings(logs)
    print(embeddings)  # These embeddings will later be used for retrieval
