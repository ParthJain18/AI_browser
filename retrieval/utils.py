import json

def load_logs(log_path):
    with open(log_path, 'r') as f:
        logs = json.load(f)
    return logs

def preprocess_logs(logs):
    preprocessed_logs = []
    for log in logs:
        chunk = f"User visited '{log['title']}' at {log['url']} with the following content: {log['text']}."
        preprocessed_logs.append(chunk)
    return preprocessed_logs

def save_metadata(processed_logs, metadata, metadata_path):
    for i, chunk in enumerate(processed_logs):
        metadata.append({"id": i, "chunk": chunk})

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)