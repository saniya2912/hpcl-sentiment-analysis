from flask import Flask, request, jsonify, render_template
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    texts = data.get("texts", [])
    results = [analyzer.polarity_scores(text) for text in texts]
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
