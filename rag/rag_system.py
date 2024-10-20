from src.retrieval.retrieval_model import search_index, index_embeddings
from src.generation.generation_model import generate_response
from src.data_processing.log_parser import load_logs, generate_embeddings

def rag_system(query_embedding, logs, embeddings):
    index = index_embeddings(embeddings)
    indices = search_index(query_embedding, index)

    # Retrieve the corresponding log entries using the indices
    retrieved_logs = [logs[i]['text'] for i in indices[0]]
    context = " ".join(retrieved_logs)  # Combine retrieved logs into a single context
    return context

if __name__ == "__main__":
    # Simulate a RAG process
    logs = load_logs('data/logs/dummy_logs.json')
    embeddings = generate_embeddings(logs)

    # Simulate a query embedding (replace with real query embedding later)
    query_embedding = embeddings[0].reshape(1, -1)  # Simulate with the first log

    # Retrieve context
    context = rag_system(query_embedding, logs, embeddings)

    # Generate response based on retrieved context
    query = "What did I read about AI?"
    response = generate_response(context, query)
    print(response)
