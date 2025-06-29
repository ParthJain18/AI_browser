from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import base64
from datetime import datetime
import os
from retrieval.utils import load_logs, preprocess_logs
from config import EMBEDDING_MODEL, VECTOR_DB_PATH, METADATA_PATH, LOGS_PATH, SCREENSHOTS_PATH
import uuid
import difflib

model = SentenceTransformer(EMBEDDING_MODEL)

def create_or_load_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        print("Loading existing vector database...")
        index = faiss.read_index(VECTOR_DB_PATH)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    else:
        print("Creating new vector database...")
        index = None
        metadata = []

        logs = load_logs(LOGS_PATH)
        preprocessed_logs = preprocess_logs(logs) if logs else None
        
        if preprocessed_logs:
            index = create_vector_db(preprocessed_logs)

    return index, metadata

def save_screenshot(screenshot_data: str, log_id: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{log_id}_{timestamp}.png"
    filepath = os.path.join(SCREENSHOTS_PATH, filename)
    
    placeholder_path = os.path.join(SCREENSHOTS_PATH, "placeholder.png")
    
    # Check if screenshot_data is valid and contains a base64 string
    if not screenshot_data or not isinstance(screenshot_data, str) or 'base64,' not in screenshot_data:
        return placeholder_path

    try:
        screenshot_data = screenshot_data.split('base64,')[1]
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(screenshot_data))
        return filepath
    except Exception as e:
        print(f"Error decoding screenshot, using placeholder: {str(e)}")
        return placeholder_path

def add_logs_to_vector_db(new_logs: list[dict], index, metadata):

    raw_logs = [{key: value for key, value in log.items() if key != 'screenshot'} for log in new_logs]
    screenshots = [log['screenshot'] for log in new_logs]
    preprocessed_logs = preprocess_logs(raw_logs)

    if not preprocessed_logs:
        print("No new logs to add.")
        return

    new_embeddings = model.encode(preprocessed_logs)

    if index is None:
        d = new_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        print("Initializing new FAISS index.")

    index.add(np.array(new_embeddings))

    new_metadata = []
    for i, (chunk, embedding, raw, screenshot) in enumerate(zip(preprocessed_logs, new_embeddings, raw_logs, screenshots)):
        url = raw.get('url')
        add_entry = True
        if url:
            existing_entries = [entry for entry in metadata if entry.get('raw', {}).get('url') == url]
            if existing_entries:
                last_entry = existing_entries[-1]
                old_description = last_entry.get('raw', {}).get('description', '')
                new_description = raw.get('description', '')
                if old_description == new_description:
                    add_entry = False
                else:
                    # Compute the diff between descriptions.
                    diff = difflib.ndiff(old_description.splitlines(), new_description.splitlines())
                    diff_text = '\n'.join(line for line in diff if line.startswith('+ '))
                    if diff_text.strip():
                        # Replace the full content with the diff text if there are changes.
                        chunk = diff_text
        if add_entry:
            log_id = str(uuid.uuid4())
            screenshot_path = save_screenshot(screenshot, log_id)
            new_metadata.append({
                'id': log_id,
                'chunk': chunk,
                'embedding': embedding.tolist(),
                'raw': raw,
                'screenshot_path': screenshot_path
            })

    if new_metadata:
        metadata.extend(new_metadata)

        faiss.write_index(index, VECTOR_DB_PATH)
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f)
        print("New logs added to the vector database.")
    else:
        print("No new logs to add – duplicate content detected.")

def create_vector_db(processed_data: list[str]):
    embeddings = model.encode(processed_data)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))

    faiss.write_index(index, VECTOR_DB_PATH)

    with open(METADATA_PATH, 'w') as f:
        json.dump([{'chunk': chunk, 'embedding': embedding.tolist()} 
                   for chunk, embedding in zip(processed_data, embeddings)], f)
    
    return index

def vectorize_query(query):
    query_embedding = model.encode([query])
    return query_embedding