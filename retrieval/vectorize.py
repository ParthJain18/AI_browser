from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from retrieval.utils import load_logs, preprocess_logs
from config import EMBEDDING_MODEL, VECTOR_DB_PATH, METADATA_PATH, LOGS_PATH
import uuid

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

    new_metadata = [{'id': str(uuid.uuid4()), 'chunk': chunk, 'embedding': embedding.tolist(), 'raw': raw, 'screenshot': screenshot} 
                    for chunk, embedding, raw, screenshot in zip(preprocessed_logs, new_embeddings, raw_logs, screenshots)]

    metadata.extend(new_metadata)

    faiss.write_index(index, VECTOR_DB_PATH)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f)

    print("New logs added to the vector database.")

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