"""SSE serialization contract tests."""

import json

import pytest

from app.generation.sse import encode_sse_event


def test_encode_sse_event_uses_json_and_blank_line_termination() -> None:
    encoded = encode_sse_event("token", {"text": "line one\nline two"})
    assert encoded.startswith("event: token\ndata: ")
    assert encoded.endswith("\n\n")
    payload = json.loads(encoded.split("data: ", 1)[1])
    assert payload == {"text": "line one\nline two"}


def test_encode_sse_event_rejects_unsafe_event_name() -> None:
    with pytest.raises(ValueError, match="event name"):
        encode_sse_event("token\nevent: fake", {})
