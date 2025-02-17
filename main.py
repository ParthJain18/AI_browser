from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import rag, add_logs
from timeline import get_user_timeline
import nltk

nltk.download('punkt_tab')
app = Flask(__name__)
CORS(app)

@app.route("/", methods= ["GET"])
def hello():
    return "Hello from server!"

@app.route("/log", methods=["POST"])
def log():
    log_data = request.json.get("entries")
    if log_data:
        print("Log recieved")
        add_logs(log_data)
    return jsonify({"status": "log received"}), 200

@app.route("/user-query", methods=["POST"])
def user_query():
    user_query = request.json.get("query")
    if user_query:
        response = rag(user_query)
        return jsonify({"response": response}), 200
    return jsonify({"error": "No query provided"}), 400

@app.route("/get-timeline", methods=["GET"])
def get_timeline():
    timeline = get_user_timeline()
    # print(timeline)
    return jsonify({"timeline": timeline}), 200

if __name__ == '__main__':
    app.run(debug=True)