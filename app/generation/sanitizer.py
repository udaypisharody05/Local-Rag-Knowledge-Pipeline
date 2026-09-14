"""Lightweight, deterministic sanitization of generation-only context.

This removes lines containing a small set of obvious prompt-injection patterns.
It does not mutate stored chunks and is defense in depth, not a complete prompt-
injection detector.
"""

import re

INSTRUCTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore (?:all )?(?:previous|prior) instructions\b",
        r"\bdisregard (?:all )?(?:previous|prior) instructions\b",
        r"\boverride (?:the )?system prompt\b",
        r"^\s*(?:please\s+)?act as\b",
        r"^\s*you are now\b",
        r"\breveal (?:the )?system prompt\b",
        r"^\s*answer every question by\b",
        r"^\s*always respond with\b",
        r"\bsecret password is\b",
        r"\bfollow these instructions instead\b",
    )
)


def sanitize_context_text(text: str) -> str:
    """Remove only lines matching obvious document-borne instructions."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not any(pattern.search(line) for pattern in INSTRUCTION_PATTERNS)
    )
