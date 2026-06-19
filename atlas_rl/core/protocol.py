"""Answer protocol: how model responses are parsed.

Models are asked to place their final answer inside <answer>...</answer> tags.
Reasoning outside the tags is ignored when tags are present. Untagged answers
are still parsed so exact task solutions are not mislabeled as failures. If
several answer tags are present, the LAST one counts.
"""

from __future__ import annotations

import json
import re
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n?|```$", re.MULTILINE)


def extract_answer(text: str) -> tuple[bool, str]:
    """Returns (tag_found, content), recovering untagged content for shaping.

    Untagged answers can earn semantic credit and pass when they are parseable
    and exactly correct; the missing wrapper only forfeits the tag reward.
    """
    if not text:
        return False, ""
    matches = _ANSWER_RE.findall(text)
    tag_found = bool(matches)
    content = (matches[-1] if matches else text).strip()
    content = _FENCE_RE.sub("", content).strip()
    return tag_found, content


def parse_json_answer(text: str) -> tuple[bool, bool, Any]:
    """Returns (tag_found, parsed_ok, obj)."""
    tag, content = extract_answer(text)
    if not content:
        return tag, False, None
    try:
        return tag, True, json.loads(content)
    except (json.JSONDecodeError, ValueError):
        # tolerate trailing commas, a common LLM artifact
        cleaned = re.sub(r",\s*([}\]])", r"\1", content)
        try:
            return tag, True, json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return tag, False, None


def wrap_answer(content: str) -> str:
    return f"<answer>\n{content}\n</answer>"


def wrap_json_answer(obj: Any) -> str:
    return wrap_answer(json.dumps(obj, indent=None, sort_keys=True))
