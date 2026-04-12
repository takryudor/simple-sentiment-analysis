from transformers import pipeline

model_id = "tabularisai/multilingual-sentiment-analysis"
_classifier = None


def get_classifier():
    """Lazy-load model to avoid heavy startup and improve resiliency."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline("sentiment-analysis", model=model_id)
    return _classifier


def is_model_loaded() -> bool:
    return _classifier is not None


def predict_sentiment(text: str):
    # Model trả về các label như: Very Negative, Negative, Neutral, Positive, Very Positive
    classifier = get_classifier()
    result = classifier(text)
    return result[0]