from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from retrieval.utils import load_logs, preprocess_logs
from config import EMBEDDING_MODEL, VECTOR_DB_PATH, METADATA_PATH, LOGS_PATH


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
        preprocessed_logs = preprocess_logs(logs)
        
        if preprocessed_logs:
            create_vector_db(preprocessed_logs)

    return index, metadata

def create_vector_db(processed_data: list[str]):
    embeddings = model.encode(processed_data)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))

    faiss.write_index(index, VECTOR_DB_PATH)

    with open(METADATA_PATH, 'w') as f:
        json.dump([{'chunk': chunk, 'embedding': embedding.tolist()} 
                   for chunk, embedding in zip(processed_data, embeddings)], f)

def vectorize_query(query):
    """
    Convert the user query into an embedding vector.
    """
    query_embedding = model.encode([query])
    return query_embedding