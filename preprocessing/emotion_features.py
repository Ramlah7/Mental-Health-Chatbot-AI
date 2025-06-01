# File: preprocessing/emotion_features.py
# ----------------------------------------------------------
# Using a pretrained, publicly available emotion detection model
# (no custom training or gated‐repo access needed).

from transformers import pipeline

# 1. Initialize a text‐classification pipeline with a publicly accessible model.
#    We have chosen "nateraw/bert-base-uncased-emotion" because it is not gated.
try:
    emotion_classifier = pipeline(
        "text-classification",
        model="nateraw/bert-base-uncased-emotion",
        return_all_scores=False
    )
    print("[emotion_features] Successfully loaded 'nateraw/bert-base-uncased-emotion'.")
except Exception as e:
    print(f"[emotion_features][ERROR] Failed to initialize emotion pipeline: {e}")
    emotion_classifier = None

# 2. Map the raw labels from "nateraw/bert-base-uncased-emotion"
#    into your target router categories.
EMOTION_MAP = {
    "anger":    "angry",
    "disgust":  "angry",    # group disgust under "angry"
    "fear":     "anxious",
    "joy":      "happy",
    "sadness":  "sad",
    "surprise": "surprised",
    "neutral":  "neutral"
}


def get_emotion(text: str) -> str:
    """
    Run the pretrained pipeline on `text` and return one of:
      {"sad", "anxious", "happy", "surprised", "loved", "angry", "neutral"}.

    Internally, the pipeline returns a raw label from the set:
      {'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'}.

    We then map that raw label into your router categories via EMOTION_MAP.
    If any error occurs or the raw label is unexpected, default to 'neutral'.
    """
    if emotion_classifier is None:
        print("[emotion_features][WARN] emotion_classifier is not initialized. Returning 'neutral'.")
        return "neutral"

    try:
        result = emotion_classifier(text)[0]
        raw_label = result.get("label", "").lower()
        score = result.get("score", None)

        print(f"[emotion_features] Input text: {text!r}")
        print(f"[emotion_features] Raw prediction: label={raw_label}, score={score}")

        mapped = EMOTION_MAP.get(raw_label)
        if mapped is None:
            print(f"[emotion_features][WARN] Unexpected raw label '{raw_label}'. Defaulting to 'neutral'.")
            return "neutral"

        print(f"[emotion_features] Mapped label: {mapped}")
        return mapped

    except Exception as e:
        print(f"[emotion_features][ERROR] Error during emotion inference: {e}")
        return "neutral"


# Optional: quick standalone test if you run this file directly
if __name__ == "__main__":
    samples = [
        "I feel wonderful today!",
        "I'm terrified of the exam tomorrow.",
        "I love spending time with my friends.",
        "This makes me disgusted.",
        "What a surprise party!",
        "I'm just okay."
    ]
    for s in samples:
        label = get_emotion(s)
        print(f"Text: {s!r} → Emotion: {label}")
