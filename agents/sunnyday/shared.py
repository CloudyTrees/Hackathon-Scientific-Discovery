"""Shared constants and utility helpers for the sunnyday agent."""

import json
import re

from hackathon_science.utils import call_llm  # noqa: F401  (re-exported for siblings)

MODEL_ID = "global.anthropic.claude-sonnet-4-6"
MAX_BODY_WORDS = 3200


def _text(response: dict) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    return content[0].get("text", "") if content else ""


def _parse_json_list(text: str) -> list[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    for candidate in [cleaned, _find_array(cleaned)]:
        if not candidate:
            continue
        try:
            v = json.loads(candidate)
            if isinstance(v, list):
                return [i for i in v if isinstance(i, dict)]
        except Exception:
            pass
    return []


def _find_array(text: str) -> str:
    m = re.search(r"\[[\s\S]*\]", text)
    return m.group(0) if m else ""


def _extract_json(output: str, start_marker: str, end_marker: str) -> dict:
    s = output.find(start_marker)
    e = output.find(end_marker)
    if s == -1 or e == -1 or e <= s:
        return {}
    try:
        v = json.loads(output[s + len(start_marker):e].strip())
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _results_table(rows: list[dict]) -> str:
    if not rows:
        return "  (no results)"
    return "\n".join(
        f"  {r.get('option_name','?')}: score={r.get('score',0)}, "
        f"acc={r.get('macro_accuracy',0)}, stab={r.get('macro_stability',0)}, "
        f"cost={r.get('macro_cost',0)}, steps={r.get('macro_steps',0)}"
        for r in rows
    )
