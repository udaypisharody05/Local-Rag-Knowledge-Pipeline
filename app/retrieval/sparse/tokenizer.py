"""Small deterministic tokenizer for prose and technical identifiers.

The tokenizer lowercases and strips surrounding punctuation while preserving
internal dashes, underscores, dots, and slash-delimited API paths. It deliberately
does not stem words or split compound identifiers, favoring exact technical-term
matching and reproducibility over linguistic recall.
"""

import re

TOKENIZER_VERSION = "technical-regex-v1"
TOKEN_PATTERN = re.compile(
    r"(?:/[a-z0-9]+(?:[-_.][a-z0-9]+)*(?:/[a-z0-9]+(?:[-_.][a-z0-9]+)*)+)"
    r"|(?:[a-z0-9]+(?:[-_./][a-z0-9]+)*)",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
