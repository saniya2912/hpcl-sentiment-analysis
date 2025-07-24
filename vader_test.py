import nltk

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create an instance of the Vader sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# List of example texts to analyze
texts = [
    "I love this product! It works great and is very affordable.",
    "This product is okay. It gets the job done, but could be better.",
    "I hate this product. It doesn't work at all and is a waste of money."
]

# Loop through the texts and get the sentiment scores for each one
for text in texts:
    scores = analyzer.polarity_scores(text)
    print(text)
    print(scores)