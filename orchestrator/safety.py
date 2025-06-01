import re

_PATTERNS = [
    re.compile(r"\bkill\s+myself\b", re.I),
    re.compile(r"\bhurt\s+myself\b", re.I),
    re.compile(r"\bsuicide\b", re.I),
]

def is_safe(text: str) -> bool:
    return not any(p.search(text) for p in _PATTERNS)