"""Run agent for sunnyday."""
from pathlib import Path
from typing import Optional

from hackathon_science import Paper
from hackathon_science.utils import call_llm

MODEL_ID = "global.anthropic.claude-sonnet-4-6"


def _ask(prompt: str) -> str:
    response = call_llm(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        model_id=MODEL_ID,
    )
    content = response.get("output", {}).get("message", {}).get("content", [])
    return content[0].get("text", "").strip() if content else ""


def run(problem_domain: str, papers_dir: Optional[Path] = None) -> Paper:
    title = _ask(
        f"You are writing a short scientific paper on: {problem_domain}\n"
        "Reply with only the paper title (no quotes, no prefix)."
    )
    introduction = _ask(
        f"Write the Introduction section for a paper titled '{title}' "
        f"on the topic: {problem_domain}. Markdown, 2-4 paragraphs."
    )
    methods = _ask(
        f"Write the Methods section for '{title}'. Be concrete about the "
        "experimental approach. Markdown, 2-3 paragraphs."
    )
    results = _ask(
        f"Write the Results section for '{title}'. Include plausible "
        "quantitative findings. Markdown, 2-3 paragraphs."
    )

    return Paper(
        title=title or f"A study of {problem_domain}",
        introduction=introduction,
        methods=methods,
        results=results,
    )
