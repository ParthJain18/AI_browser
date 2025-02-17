from retrieval.vectorize import vectorize_query, create_or_load_vector_db, add_logs_to_vector_db
from retrieval.retrieval_model import retrieve_relevant_chunks
from generation.generation import get_response
from utils.screenshot_util import load_screenshot_from_path
from config import METADATA_PATH
import json

index, metadata = create_or_load_vector_db()

def load_metadata():
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    return metadata

def rag(user_query):
    metadata = load_metadata()
    query_embedding = vectorize_query(user_query)
    relevant_logs = retrieve_relevant_chunks(query_embedding, index=index, metadata=metadata, top_k=3)
    context = " ".join([log['chunk'] for log in relevant_logs])
    response = get_response(user_query, context)

    # print(metadata[0].keys())

    sources = [
        {
            'id': log['id'],
            'raw': next(item['raw'] for item in metadata if item['id'] == log['id']),
            'screenshot': next(load_screenshot_from_path(item['screenshot_path']) for item in metadata if item['id'] == log['id'])
        }
        for log in relevant_logs
    ]
    return response, sources

def add_logs(logs):
    add_logs_to_vector_db(logs, index, metadata)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    rag(user_query)