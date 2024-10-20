import faiss
import json
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_relevant_chunks(user_query):
    # Load vector database
    index = faiss.read_index("retrieval/vector_db/index.faiss")

    # Convert user query into an embedding
    query_embedding = model.encode([user_query])

    # Search for the closest vectors
    D, I = index.search(np.array(query_embedding), k=5)  # Retrieve top 5 matches

    # Load metadata to get chunk details
    with open("retrieval/vector_db/vector_metadata.json", 'r') as f:
        metadata = json.load(f)

    # Return the chunks corresponding to the top matches
    return [metadata[i]['chunk'] for i in I[0]]
