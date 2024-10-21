from groq import Groq
from config import GROQ_API_KEY

client = Groq(
    api_key=GROQ_API_KEY,
)

def get_response(data, context):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""You are a RAG agent. Your job is to answer user queries regarding their browsing activities. You must respond in a JSON format of this structure:
                        {{
                            "response":"text"
                        }}

                    These are the three most recent user activities. They may or may not be useful to answer the user's query. If the user's query can be answered by the retrieved documents, answer them, otherwise, reply that the system couldn't find a matching activity.

                    Fetched activities are:
                    {context}

                """
            },
            {
                "role": "user",
                "content": data,
            }
        ],
        model="llama3-8b-8192",
    )

    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content