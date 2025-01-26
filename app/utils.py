from transformers import pipeline
import logging
from datetime import datetime

# Initialize sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def sanitize_input(user_input):
    """
    Sanitizes user input to prevent unintended issues with the AI model.
    """
    return user_input.strip().replace("\n", " ")

def setup_logging():
    """
    Sets up logging for the application.
    """
    import os
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"),
            logging.StreamHandler()
        ]
    )

def log_error(message):
    """
    Logs errors with a timestamp.
    """
    logging.error(message)

def log_info(message):
    """
    Logs general information with a timestamp.
    """
    logging.info(message)

def detect_sentiment(user_input):
    """
    Detects sentiment using a pre-trained Hugging Face model.
    Categorizes input as positive, neutral, or negative.
    """
    try:
        analysis = sentiment_analyzer(user_input)
        sentiment_label = analysis[0]["label"]  # "POSITIVE" or "NEGATIVE"
        sentiment_score = analysis[0]["score"]  # Confidence score

        if sentiment_label == "POSITIVE" and sentiment_score > 0.8:
            return "positive"
        elif sentiment_label == "NEGATIVE" and sentiment_score > 0.8:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        log_error(f"Error detecting sentiment: {e}")
        return "neutral"
