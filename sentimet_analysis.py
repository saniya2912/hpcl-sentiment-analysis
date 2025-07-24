import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# Option 1: Manual multi-line input
print("Enter multiple lines of text (press Enter twice to finish):")
lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)

# Option 2: Load from text file (uncomment below if needed)
# with open("input.txt", "r") as file:
#     lines = [line.strip() for line in file.readlines() if line.strip()]

# Analyze sentiment
results = [analyzer.polarity_scores(text) for text in lines]

# Print results
for i, (text, score) in enumerate(zip(lines, results), 1):
    print(f"\nLine {i}: {text}")
    print(score)

# Prepare data for charts
positives = [r['pos'] for r in results]
neutrals = [r['neu'] for r in results]
negatives = [r['neg'] for r in results]
labels = [f"Line {i+1}" for i in range(len(lines))]

# --- Pie Chart (Average sentiment) ---
avg_pos = sum(positives) / len(positives)
avg_neu = sum(neutrals) / len(neutrals)
avg_neg = sum(negatives) / len(negatives)

plt.figure(figsize=(6, 6))
plt.pie([avg_pos, avg_neu, avg_neg],
        labels=['Positive', 'Neutral', 'Negative'],
        colors=['green', 'gray', 'red'],
        autopct='%1.1f%%',
        startangle=140)
plt.title('Average Sentiment Breakdown (Pie Chart)')
plt.axis('equal')
plt.show()

# --- Line Chart (Sentiment per line) ---
plt.figure(figsize=(10, 5))
plt.plot(labels, positives, label='Positive', color='green', marker='o')
plt.plot(labels, neutrals, label='Neutral', color='gray', marker='o')
plt.plot(labels, negatives, label='Negative', color='red', marker='o')
plt.title('Sentiment Score per Line')
plt.xlabel('Text Lines')
plt.ylabel('Sentiment Score')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
