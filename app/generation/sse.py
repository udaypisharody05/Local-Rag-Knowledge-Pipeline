"""Safe serialization for the small query-stream SSE protocol."""

import json
import re

EVENT_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


def encode_sse_event(event_name: str, payload: object) -> str:
    if not EVENT_NAME.fullmatch(event_name):
        raise ValueError("Invalid SSE event name")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data}\n\n"
