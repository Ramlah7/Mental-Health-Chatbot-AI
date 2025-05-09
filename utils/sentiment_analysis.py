# sentiment_analysis.py
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK corpora (only once)
nltk.download('punkt')

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyzes the sentiment of a given text and returns Positive, Negative, or Neutral."""
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Example usage
if __name__ == "__main__":
    test_text = "I'm feeling very stressed and overwhelmed."
    sentiment = analyze_sentiment(test_text)
    print(f"Sentiment: {sentiment}")
