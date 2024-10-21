import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def load_logs(log_path):
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
        return logs
    except json.JSONDecodeError:
        print(f"Warning: The log file at {log_path} is empty or contains invalid JSON.")
        return []

def filter_text(text): 
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    filtered_text = ' '.join(filtered_text)
    return filtered_text

def preprocess_logs(logs):
    preprocessed_logs = []
    for log in logs:
        title = log['title']
        content = filter_text(log['summary'])
        description = filter_text(log['description'])
        keywords = str(log['keywords'])

        text = f"""User visited "{title}" with the following content: "{content}". \n\n The site had the following description: {description} \n\n Keywordsm from the site are: {keywords}"""
        preprocessed_logs.append(text)
    
    print(preprocessed_logs)

    return preprocessed_logs