# from transformers import pipeline
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, resources={r"/detect-bias": {"origins": "http://127.0.0.1:5500"}})


# # Load the sentiment analysis model
# nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased")


# @app.route("/detect-bias", methods=["POST"])
# def detect_bias():
#     texts = request.json.get("texts")  # Expecting 'texts' to be an array of strings

#     if not isinstance(texts, list) or not texts:
#         return jsonify({"error": "Invalid input. Please provide a list of texts."}), 400

#     # Process each text and collect results
#     results = [{"text": text, "sentiment": nlp(text)} for text in texts]

#     return jsonify(results)


# if __name__ == "__main__":
#     app.run(debug=True, port=5000)


from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/detect-bias": {"origins": "http://127.0.0.1:5500"}})

# Load BERT model for text classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


@app.route("/detect-bias", methods=["POST"])
def detect_bias():
    data = request.get_json()
    texts = data.get("texts", [])  # Expecting 'texts' to be an array of strings

    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Invalid input. Please provide a list of texts."}), 400

    # Define labels for zero-shot classification
    candidate_labels = ["gender bias", "neutral", "positive", "negative"]

    # Process each text and collect results
    results = []
    for text in texts:
        result = classifier(text, candidate_labels)
        results.append({"text": text, "classification": result})

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
