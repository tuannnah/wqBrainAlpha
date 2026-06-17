"""Parse JSON an toàn từ output LLM (gỡ fences, tìm khối {..}/[..] đầu tiên)."""

from __future__ import annotations

import json


def extract_json(content: str) -> dict | list | None:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except (json.JSONDecodeError, ValueError):
                    continue
    return None
