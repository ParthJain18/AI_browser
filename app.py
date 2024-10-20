import json
import os
import faiss
import numpy as np
from sentence_transformer import SentenceTransformer
from data_processing.log_parser import preprocess_logs
from utils import load_logs, save_metadata, generate_response_from_llm


VECTOR_DB_PATH = "data/vector_db/index.faiss"
METADATA_PATH = "data/vector_db/vector_metadata.json"
LOGS_PATH = "data/logs/browsing_log.json"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def create_or_load_vector_db():

    model = SentenceTransformer(EMBEDDING_MODEL)

    if os.path.exists(VECTOR_DB_PATH):
        print("Loading existing vector database...")
        index = faiss.read_index(VECTOR_DB_PATH)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    else:
        print("Creating new vector database...")
        index = None
        metadata = []

        logs = load_logs(LOGS_PATH)  # Loads raw logs from the file
        preprocessed_logs = preprocess_logs(logs)  # Preprocess logs into text chunks
        
        if preprocessed_logs:
            # Convert logs to embeddings
            embeddings = model.encode(preprocessed_logs)
            
            # Create FAISS index and add embeddings
            d = embeddings.shape[1]  # Dimension of embeddings
            index = faiss.IndexFlatL2(d)  # L2 distance-based FAISS index
            index.add(np.array(embeddings))
            
            # Save the vector database and metadata
            faiss.write_index(index, VECTOR_DB_PATH)
            save_metadata(preprocessed_logs, metadata, METADATA_PATH)  # Save preprocessed logs with metadata

    return index, metadata


def vectorize_query(query):
    """
    Convert the user query into an embedding vector.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])
    return query_embedding


def retrieve_relevant_logs(query_embedding, index, metadata, top_k=5):
    """
    Retrieve the most relevant logs from the vector database based on the query embedding.
    """
    D, I = index.search(np.array(query_embedding), k=top_k)
    
    # Get corresponding logs from metadata
    retrieved_logs = [metadata[i]['chunk'] for i in I[0]]
    return retrieved_logs


def main():
    # Step 1: Create or load vector database
    index, metadata = create_or_load_vector_db()

    # Step 2: Get user query input
    user_query = input("Enter your query: ")

    # Step 3: Vectorize user query
    query_embedding = vectorize_query(user_query)

    # Step 4: Retrieve relevant logs based on query
    relevant_logs = retrieve_relevant_logs(query_embedding, index, metadata)

    # Step 5: Generate response using retrieved logs and user query
    context = " ".join(relevant_logs)  # Combine relevant log chunks as context
    response = generate_response_from_llm(user_query, context)

    # Display the response
    print("Generated Response:\n", response)


if __name__ == "__main__":
    main()
