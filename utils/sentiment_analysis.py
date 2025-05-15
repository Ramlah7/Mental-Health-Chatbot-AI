# sentiment_analysis.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np
import re
import emoji

# Load the model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Labels mapping
LABELS = ['Negative', 'Neutral', 'Positive']

def preprocess(text):
    """Cleans tweet text to match training data expectations."""
    # Convert emojis and remove special symbols
    text = emoji.demojize(text)
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)     # remove @mentions
    text = re.sub(r'#', '', text)        # remove hashtags symbol only
    text = text.replace('\n', ' ').strip()
    return text

def analyze_sentiment(text):
    """Analyzes sentiment using a RoBERTa model fine-tuned on tweets."""
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = F.softmax(output.logits, dim=1)
    ranking = torch.argmax(scores).item()
    sentiment = LABELS[ranking]
    return sentiment

# Example usage
if __name__ == "__main__":
    test_text = "I'm feeling super stressed and anxious 😞"
    print(f"Sentiment: {analyze_sentiment(test_text)}")
