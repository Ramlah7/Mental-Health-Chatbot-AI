"""Router with simple pattern-based replies for greetings, name queries, thanks, self-harm filtering, FAQ lookup, and generator fallback."""

from preprocessing.text_normalizer import normalize
from retrieval.index import faq_query
from chatbot.rule_based_chatbot import generate_bot_reply as generate_reply
import traceback
import re

# Similarity threshold for FAQ matching
SIM_THRESHOLD = 0.78
# Number of sentences to keep from a generated reply
MAX_SENTENCES = 5

# Simple pattern-response map
PATTERN_RESPONSES = [
    (r"^(hi|hello|hey|yo|good morning|good evening|good afternoon)$", "Hello! I'm MindMate, your support buddy. How can I help you today?"),
    (r"what('?s| is) your name", "I'm MindMate! I'm here to listen and support you."),
    (r"thank(s| you)", "You're welcome! I'm always here if you need to talk."),
]

# Keywords to detect self-harm or violence ideation
SELF_HARM_KEYWORDS = [
    "suicide", "kill myself", "end it all", "kill me", "die"
]

# Predefined safe response for self-harm ideation including helpline numbers
SELF_HARM_RESPONSE = (
    "I’m really sorry you’re feeling so distressed. You deserve help and support. "
    "Please reach out right now to someone you trust or a mental health professional.\n"
    "Here are helpline numbers you can call for immediate support:\n"
    "Umang (Nationwide 24/7 Mental Health & Suicide Prevention Helpline)\n"
    "Phone (Mobile/WhatsApp): +92 311 778 6264 / +92 311 (77 UMANG)\n"
    "Rozan Counseling Helpline (Psychosocial Support for Emotional & Psychological Concerns)\n"
    "Phone (Toll-Free): 0800-22444 / Phone (Mobile): +92 303 444 2288"
)


def _safe(func, *args, fallback=None, tag="step"):
    try:
        return func(*args)
    except Exception as e:
        print(f"[router][ERROR] {tag}: {e}")
        traceback.print_exc()
        return fallback


def truncate_reply_text(text: str, max_sentences: int = MAX_SENTENCES) -> str:
    """Shorten the text to the first max_sentences sentences."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    if not sentences:
        return text
    truncated = " ".join(sentences[:max_sentences])
    if not re.search(r'[.!?]$', truncated):
        truncated = truncated.rstrip() + '...'
    return truncated


def route(user_raw: str) -> str:
    """Main routing function: simple regex patterns, self-harm filter, FAQ lookup, then generator fallback."""
    # 1. Normalize user input
    user_norm = normalize(user_raw)
    print(f"[router] RAW → {user_raw}")
    print(f"[router] NORM → {user_norm}")

    lowered = user_norm.strip().lower()

    # 2. Check simple pattern-based responses
    for pattern, reply in PATTERN_RESPONSES:
        if re.search(pattern, lowered):
            print(f"[router] PATTERN matched: {pattern}")
            return reply

    # 3. Self-harm keyword detection
    for kw in SELF_HARM_KEYWORDS:
        if kw in lowered:
            print(f"[router] SELF-HARM detected keyword: '{kw}'")
            return SELF_HARM_RESPONSE

    # 4. FAQ lookup for definition-like or common queries
    result = _safe(faq_query, user_norm, fallback=None, tag="faq_query")
    candidate, sim = None, 0.0
    if isinstance(result, tuple) and len(result) == 2:
        candidate, sim = result
    elif isinstance(result, str):
        candidate = result
        sim = 1.0
    print(f"[router] FAQ sim={sim:.2f} | hit={candidate}")
    if candidate and sim >= SIM_THRESHOLD:
        return truncate_reply_text(candidate, MAX_SENTENCES)

    # 5. Generator fallback (truncated to max sentences)
    full_reply = _safe(generate_reply, user_raw, fallback="I'm not sure how to respond.", tag="generator")
    if not full_reply:
        return "I'm not sure how to respond."
    short_reply = truncate_reply_text(full_reply, MAX_SENTENCES)
    return short_reply
