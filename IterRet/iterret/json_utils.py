
from __future__ import annotations

import json
import re
from typing import Any, Dict

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_object(raw: str) -> Dict[str, Any]:
    if not raw or not isinstance(raw, str):
        return {}

    candidates = [raw.strip()]

    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    # Fall back to the widest brace-delimited substring (handles preamble /
    # trailing commentary around an otherwise-valid object).
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}
