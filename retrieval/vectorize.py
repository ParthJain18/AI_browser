from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

model = SentenceTransformer('all-MiniLM-L6-v2')


def create_vector_db(processed_data: list[str]):
    embeddings = model.encode([chunk['text'] for chunk in processed_data])

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))

    faiss.write_index(index, "retrieval/vector_db/index.faiss")

    with open("retrieval/vector_db/vector_metadata.json", 'w') as f:
        json.dump([{'chunk': chunk, 'embedding': embedding.tolist()} 
                   for chunk, embedding in zip(processed_data, embeddings)], f)
