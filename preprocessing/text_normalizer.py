"""Updated text_normalizer.py to use SymSpell with your frequency dictionary for spell correction."""

import os
import re
from symspellpy.symspellpy import SymSpell, Verbosity

# Maximum edit distance for corrections
MAX_EDIT_DISTANCE = 2
# Prefix length for SymSpell
PREFIX_LENGTH = 7

# Filename of your frequency dictionary in the data folder
DICTIONARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'frequency_dictionary_en_82_765.txt')
# If your file has a different name, update DICTIONARY_PATH accordingly

# Pattern to remove non-alphanumeric characters (except spaces)
CLEANR = re.compile(r"[^\w\s]")

# Initialize SymSpell once
sym_spell = SymSpell(MAX_EDIT_DISTANCE, PREFIX_LENGTH)
if os.path.exists(DICTIONARY_PATH):
    # Load frequency dictionary: expects tab-delimited "term\tcount"
    sym_spell.load_dictionary(DICTIONARY_PATH, term_index=0, count_index=1)
else:
    print(f"[text_normalizer] WARNING: Dictionary file not found at {DICTIONARY_PATH}")


def normalize(text: str) -> str:
    """Normalize input by lowercasing, removing punctuation, and using SymSpell to correct spelled words."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(CLEANR, "", text)
    # Tokenize
    tokens = text.split()
    corrected_tokens = []
    for token in tokens:
        # Lookup suggestions with maximum edit distance
        suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=MAX_EDIT_DISTANCE)
        if suggestions:
            # Take the first suggestion (highest term frequency)
            corrected_tokens.append(suggestions[0].term)
        else:
            corrected_tokens.append(token)
    # Reassemble corrected text
    return " ".join(corrected_tokens)

# Example usage:
# If "helo" appears, sym_spell will suggest "hello" if in dictionary.
# normalize("Helo wrld!") -> "hello world"
