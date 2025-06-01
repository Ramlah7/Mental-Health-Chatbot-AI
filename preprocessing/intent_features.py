# File: preprocessing/intent_features.py
# ----------------------------------------------------------
# Using Hugging Face zero-shot classifier (facebook/bart-large-mnli)
# to predict intent labels. No custom training needed.

from transformers import pipeline
# Note: Removed "PipelineException" import since it’s not available in this TF version.

# List of intent labels as defined
INTENT_LABELS = [
    "coping_advice",
    "definition_request",
    "greeting",
    "label",
    "other",
    "self_harm",
    "small_talk"
]

# Confidence threshold for labeling as other if below
INTENT_CONF_THRESHOLD = 0.40

# Create a zero-shot classification pipeline (wrapped in try/except)
try:
    intent_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    print("[intent_features] Successfully loaded 'facebook/bart-large-mnli' pipeline.")
except Exception as e:
    print(f"[intent_features][ERROR] Failed to initialize intent pipeline: {e}")
    intent_classifier = None


def get_intent(text: str):
    """
    Returns a tuple (label, confidence_score).
    If confidence_score < threshold, returns ("other", confidence_score).
    """
    if intent_classifier is None:
        print("[intent_features][WARN] intent_classifier is not initialized. Returning ('other', 0.0).")
        return "other", 0.0

    try:
        print(f"[intent_features] Input text: {text!r}")
        result = intent_classifier(text, candidate_labels=INTENT_LABELS)

        # result["labels"] is a list of labels sorted by descending score
        # result["scores"] is the corresponding list of scores
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])

        print(f"[intent_features] Raw predictions: {result['labels']} with scores {result['scores']}")
        print(f"[intent_features] Top label: {top_label!r}, Score: {top_score:.4f}")

        # If below threshold, force to 'other'
        if top_score < INTENT_CONF_THRESHOLD:
            print(f"[intent_features] Score {top_score:.4f} < threshold {INTENT_CONF_THRESHOLD}. Forcing 'other'.")
            return "other", top_score

        return top_label, top_score

    except Exception as e:
        # Catch any exception during inference and default to ("other", 0.0)
        print(f"[intent_features][ERROR] Error during intent inference: {e}")
        return "other", 0.0



if __name__ == "__main__":
    samples = [
        "How do I cope with anxiety?",
        "Hello there!",
        "What does 'biometrics' mean?",
        "I feel like hurting myself.",
        "Just chatting."
        "what is anxiety?"
    ]
    for s in samples:
        lbl, sc = get_intent(s)
        print(f"Text: {s!r} → Intent: {lbl}, Confidence: {sc:.4f}")
