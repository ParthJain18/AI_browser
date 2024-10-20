from retrieval.vectorize import vectorize_query, create_or_load_vector_db
from retrieval.retrieval_model import retrieve_relevant_chunks
from generation.utils import generate_response_from_llm

def main():
    index, metadata = create_or_load_vector_db()

    user_query = input("Enter your query: ")
    query_embedding = vectorize_query(user_query)
    relevant_logs = retrieve_relevant_chunks(query_embedding, index=index, metadata=metadata, top_k=1)
    context = " ".join(relevant_logs)
    response = generate_response_from_llm(user_query, context)

    print("Generated Response:\n", response)

if __name__ == "__main__":
    main()