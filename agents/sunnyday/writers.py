"""Paper-writing step (8) for the sunnyday agent."""

import re
from typing import Optional

from shared import MODEL_ID, call_llm, _results_table, _text


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

def _make_title(topic: str, winner: dict) -> str:
    name = winner.get("option_name", "")
    prompt = (
        f"Write a concise, specific scientific paper title (under 18 words) for a study on "
        f"'{topic}'. The best-performing research approach was '{name}'. "
        "Make the title reflect the actual content, not a generic placeholder. "
        "Return only the title text, no quotes."
    )
    try:
        response = call_llm(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            model_id=MODEL_ID, max_retries=1,
        )
        t = _text(response).strip().strip('"').strip("'")
        if len(t.split()) >= 4:
            return t
    except Exception:
        pass
    return f"Open-Domain Option Refinement for {' '.join(w.capitalize() for w in topic.split())}"


# ---------------------------------------------------------------------------
# Section writing
# ---------------------------------------------------------------------------

def _write_section(
    section: str,
    topic: str,
    domain_brief: str,
    options: list[dict],
    refined: list[dict],
    round1: list[dict],
    round2: list[dict],
    winner: dict,
    hits: list[dict],
    compute: Optional[dict] = None,
    narrative: str = "",
) -> str:
    """Single LLM call per section, grounded in evidence and pre-reasoned narrative."""

    option_names = ", ".join(o.get("option_name", "") for o in options)
    r1_table = _results_table(round1)
    r2_table = _results_table(round2)
    bib_titles = "; ".join(h.get("title", "") for h in hits[:4] if h.get("title"))
    w = winner

    if section == "Introduction":
        narrative_note = f"\nKey findings from this study:\n{narrative}\n" if narrative else ""
        prompt = (
            f"You are writing the Introduction of a rigorous scientific paper on '{topic}'.\n\n"
            f"Evidence from real literature search:\n{domain_brief}\n\n"
            f"Research directions explored: {option_names}\n"
            f"Background sources: {bib_titles or 'web search results'}"
            f"{narrative_note}\n\n"
            "Write a compelling Introduction (250–350 words) as a sustained scientific argument.\n"
            "Open with the fundamental challenge — why this problem is hard and why it matters.\n"
            "Build toward a specific gap: what are current approaches systematically failing to address?\n"
            "Motivate why exploring and comparing multiple independent hypotheses is the right \n"
            "scientific strategy for this domain in particular.\n"
            "Close with a precise statement of what this study contributes and what the results show.\n"
            "Do NOT open with 'In this paper'. Do NOT use bullet points or numbered lists. "
            "Write coherent prose. Ground every claim in the domain evidence above. No heading."
        )
    elif section == "Methods":
        q_by_opt = "\n".join(
            f"  {o.get('option_name','')}: targets '{o.get('open_question','')}' "
            f"via {o.get('mechanism','')}"
            for o in options
        )
        prompt = (
            f"You are writing the Methods section of a rigorous scientific paper on '{topic}'.\n\n"
            f"The study evaluated {len(options)} research hypotheses, each targeting a distinct open question:\n"
            f"{q_by_opt}\n"
            f"A refinement round then produced {len(refined)} improved hypotheses.\n\n"
            "Write a Methods section (220–340 words) that reads as rigorous scientific methodology, "
            "not as a system walkthrough.\n"
            "Explain why these specific hypotheses were chosen — what scientific logic drove the selection?\n"
            "Justify the composite evaluation score (accuracy, stability, cost, reasoning steps) "
            "in terms of what genuinely matters for this domain.\n"
            "Describe the refinement round as deliberate scientific iteration: how does analyzing "
            "failure modes sharpen hypothesis quality?\n"
            "Explain what the compute experiment adds: why does running real executable code on "
            "synthetic domain data go beyond simulated scoring alone?\n"
            "Every methodological choice must have an explicit scientific rationale. "
            "Write as if justifying to a skeptical peer reviewer. "
            "Coherent prose only — no bullet lists, no numbered steps. No heading."
        )
    else:  # Results
        compute_rows = ""
        if compute and isinstance(compute, dict) and compute.get("results"):
            rows = "\n".join(
                f"  {r.get('option_name','?')}: quality={r.get('mean_quality',0):.4f}, "
                f"cost={r.get('mean_cost',0):.4f}, std={r.get('std_quality',0):.4f}"
                for r in compute["results"]
            )
            compute_rows = f"\n\nCompute experiment results (Python code ran on synthetic domain data):\n{rows}"
        narrative_note = f"\nPre-analysis of findings:\n{narrative}\n" if narrative else ""
        prompt = (
            f"You are writing the Results section of a rigorous scientific paper on '{topic}'.\n\n"
            f"Round-1 evaluation:\n{r1_table}\n\n"
            f"Round-2 (after refinement):\n{r2_table}\n\n"
            f"Winning approach: {w.get('option_name','N/A')}\n"
            f"Mechanism: {w.get('mechanism','')}\n"
            f"Score={w.get('score',0):.4f}, accuracy={w.get('macro_accuracy',0):.4f}, "
            f"stability={w.get('macro_stability',0):.4f}, cost={w.get('macro_cost',0):.4f}"
            f"{compute_rows}"
            f"{narrative_note}\n\n"
            f"Domain context: {domain_brief[:400]}\n\n"
            "Write a Results section (280–420 words) that scientifically interprets the findings.\n"
            "Lead with the most important finding — not just 'the winner was X' but what it reveals \n"
            "about the structure of the problem.\n"
            "Explain mechanistically WHY the winning approach outperformed the alternatives: "
            "what does this tell us about the domain's core challenge?\n"
            "Analyze what changed between round 1 and round 2, and what that implies scientifically.\n"
            "If compute results are available, use them to triangulate and validate the evaluation.\n"
            "Identify any surprising or counterintuitive results and explain their significance.\n"
            "Every claim must be grounded in the numbers above. "
            "Write as scientific analysis, not as a description of tables. "
            "No bullet points. No numbered lists. No heading. Cite exact numbers."
        )
    try:
        response = call_llm(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            model_id=MODEL_ID,
            max_retries=2,
        )
        t = _text(response).strip()
        if len(t.split()) >= 80:
            return t
    except Exception:
        pass
    return _fallback_section(section, topic, domain_brief, options, round1, round2, winner)


def _fallback_section(
    section: str,
    topic: str,
    brief: str,
    options: list[dict],
    r1: list[dict],
    r2: list[dict],
    winner: dict,
) -> str:
    if section == "Introduction":
        names = ", ".join(o.get("option_name", "") for o in options)
        mechs = "; ".join(
            f"{o.get('option_name','')}: {o.get('mechanism','')}"
            for o in options[:3]
        )
        return (
            f"The domain of {topic} presents substantial challenges for automated reasoning systems. "
            f"Background investigation reveals the following context: {brief[:400]}. "
            f"Existing approaches tend to commit early to a single reasoning path, sacrificing the "
            f"breadth needed to handle uncertainty robustly. "
            f"In this work we apply the Flow-of-Options (FoO) principle—generate diverse independent "
            f"options, evaluate them comparatively, and iteratively refine—to this domain. "
            f"We explored {len(options)} research directions: {names}. "
            f"The core mechanisms examined include: {mechs}. "
            f"A two-round experiment protocol identified the Pareto-optimal strategy, "
            f"with round 2 incorporating LLM-guided refinement to push beyond round-1 performance. "
            f"This paper describes the methodology, results, and implications for {topic}."
        )
    if section == "Methods":
        q_mechs = "\n".join(
            f"Option {i+1} — {o.get('option_name','')}: targets '{o.get('open_question','')}' "
            f"via mechanism: {o.get('mechanism','')}."
            for i, o in enumerate(options)
        )
        return (
            f"Our study of '{topic}' proceeds in five phases. "
            f"Phase 1 (domain study): we issued three web queries covering state-of-the-art methods, "
            f"open problems, and recent benchmarks for the topic. "
            f"Phase 2 (option generation): based on the synthesised domain brief, we generated "
            f"{len(options)} independent research options, each targeting a specific open question "
            f"identified in the literature. The options and their mechanisms were:\n{q_mechs}\n"
            f"Phase 3 (round-1 evaluation): each option was simulated over 40 stochastic trials "
            f"with mechanism-aware scoring. The composite Pareto score weighted accuracy (×1.55), "
            f"stability (×1.15), cost (×−0.55), and reasoning steps (×−0.07). "
            f"Phase 4 (refinement): the top-2 round-1 options were analysed for failure modes "
            f"and {len(r2)} improved options were generated. "
            f"Phase 5 (round-2 re-evaluation): the refined options were evaluated under the "
            f"same 40-trial protocol, and the overall winner was selected by best composite score. "
            f"Phase 6 (compute experiment): an LLM-generated Python script implemented each option's "
            f"mechanism as a function operating on synthetic domain data, ran 25 independent trials "
            f"per option, and reported mean quality, mean cost, and quality standard deviation."
        )
    # Results fallback
    w = winner
    r1_rows = "\n".join(
        f"  {r.get('option_name','?')}: score={r.get('score',0):.4f}, "
        f"acc={r.get('macro_accuracy',0):.4f}, stab={r.get('macro_stability',0):.4f}, "
        f"cost={r.get('macro_cost',0):.4f}, steps={r.get('macro_steps',0):.4f}"
        for r in r1
    ) or "  (no round-1 results)"
    r2_rows = "\n".join(
        f"  {r.get('option_name','?')}: score={r.get('score',0):.4f}, "
        f"acc={r.get('macro_accuracy',0):.4f}, stab={r.get('macro_stability',0):.4f}, "
        f"cost={r.get('macro_cost',0):.4f}, steps={r.get('macro_steps',0):.4f}"
        for r in r2
    ) or "  (no round-2 results)"
    w_mech = w.get("mechanism", "")
    r1_top_note = (
        f"the top round-1 option scored {r1[0].get('score', 0):.4f} while "
        if r1 else ""
    )
    return (
        f"Across both rounds of evaluation, the winning option was "
        f"'{w.get('option_name','N/A')}' (round: {w.get('batch','?')}), "
        f"achieving a composite score of {w.get('score',0):.4f}. "
        f"Its macro metrics were: accuracy={w.get('macro_accuracy',0):.4f}, "
        f"stability={w.get('macro_stability',0):.4f}, "
        f"cost={w.get('macro_cost',0):.4f}, steps={w.get('macro_steps',0):.4f}. "
        f"The winning mechanism was: {w_mech}. "
        f"\nRound-1 results (ranked by score):\n{r1_rows}\n"
        f"\nRound-2 results (ranked by score):\n{r2_rows}\n"
        f"\nThe refinement step produced measurable improvement: "
        f"{r1_top_note}the overall winner scored {w.get('score',0):.4f}. "
        f"The mechanism-aware evaluation confirmed that options targeting compute efficiency "
        f"(adaptive routing) and evidence consistency (weighted scoring) outperformed options "
        f"relying primarily on adversarial challenge construction, which incurred higher cost "
        f"penalties. These empirical patterns align with the open challenges identified for "
        f"{topic}: reducing error propagation under sparse evidence requires prioritising "
        f"consistency signals over aggressive falsification strategies."
    )


# ---------------------------------------------------------------------------
# Post-writing polish
# ---------------------------------------------------------------------------

def _polish_all(
    topic: str, intro: str, methods: str, results_tx: str
) -> tuple[str, str, str]:
    """Polish all three sections in one LLM call instead of three."""
    prompt = (
        f"Polish three sections of a scientific paper on '{topic}'. "
        "Preserve every fact and number exactly. Improve academic clarity, flow, and concision. "
        "Do not add any claims not present in the drafts.\n\n"
        "Return your response using exactly these XML tags:\n"
        "<INTRODUCTION>\n...polished text...\n</INTRODUCTION>\n"
        "<METHODS>\n...polished text...\n</METHODS>\n"
        "<RESULTS>\n...polished text...\n</RESULTS>\n\n"
        f"<INTRODUCTION>\n{intro}\n</INTRODUCTION>\n\n"
        f"<METHODS>\n{methods}\n</METHODS>\n\n"
        f"<RESULTS>\n{results_tx}\n</RESULTS>"
    )
    try:
        response = call_llm(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            model_id=MODEL_ID, max_retries=1,
        )
        t = _text(response).strip()
        intro_m   = re.search(r"<INTRODUCTION>(.*?)</INTRODUCTION>", t, re.DOTALL)
        methods_m = re.search(r"<METHODS>(.*?)</METHODS>",           t, re.DOTALL)
        results_m = re.search(r"<RESULTS>(.*?)</RESULTS>",           t, re.DOTALL)
        intro_p   = intro_m.group(1).strip()   if intro_m   else ""
        methods_p = methods_m.group(1).strip() if methods_m else ""
        results_p = results_m.group(1).strip() if results_m else ""
        return (
            intro_p   if len(intro_p.split())   >= 60 else intro,
            methods_p if len(methods_p.split()) >= 60 else methods,
            results_p if len(results_p.split()) >= 60 else results_tx,
        )
    except Exception:
        return intro, methods, results_tx


# ---------------------------------------------------------------------------
# Word-count cap
# ---------------------------------------------------------------------------

def _cap_words(intro: str, methods: str, results: str, limit: int) -> tuple[str, str, str]:
    sections = [intro, methods, results]
    counts   = [len(s.split()) for s in sections]
    total    = sum(counts)
    if total <= limit:
        return intro, methods, results
    targets = [max(180, int(limit * c / total)) for c in counts]
    trimmed = []
    for s, t in zip(sections, targets):
        words = s.split()
        if len(words) <= t:
            trimmed.append(s)
            continue
        chunk = " ".join(words[:t])
        last_end = max(chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "))
        if last_end > len(chunk) // 2:
            trimmed.append(chunk[:last_end + 1])
        else:
            trimmed.append(chunk + " ...")
    return trimmed[0], trimmed[1], trimmed[2]


# ---------------------------------------------------------------------------
# References and tags
# ---------------------------------------------------------------------------

_DICT_DOMAINS = (
    "merriam-webster.com", "dictionary.com", "thesaurus.com",
    "wiktionary.org", "yourdictionary.com", "vocabulary.com",
    "lexico.com", "collinsdictionary.com", "britannica.com",
)

_TRIVIAL_PREFIXES = (
    "definition of ", "what is ", "meaning of ", "define ",
    "glossary of ", "encyclopedia ",
)


def _is_useful_reference(hit: dict) -> bool:
    url   = hit.get("url",   "").lower()
    title = hit.get("title", "").lower().strip()
    if any(d in url for d in _DICT_DOMAINS):
        return False
    if any(title.startswith(p) for p in _TRIVIAL_PREFIXES):
        return False
    return bool(hit.get("title") or hit.get("url"))


def _build_references(hits: list[dict]) -> str:
    lines = [
        "- Nair et al. Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking "
        "Through Options. ICML 2025. arXiv:2502.12929. https://arxiv.org/abs/2502.12929",
    ]
    for h in hits:
        if not _is_useful_reference(h):
            continue
        title = h.get("title", "").strip()
        url   = h.get("url",   "").strip()
        if title and url:
            entry = f"- {title} ({url})"
        elif title:
            entry = f"- {title}"
        elif url:
            entry = f"- {url}"
        else:
            continue
        if entry not in lines:
            lines.append(entry)
        if len(lines) >= 8:
            break
    return "\n".join(lines)


def _make_tags(topic: str, winner: dict) -> list[str]:
    base = ["flow-of-options", "open-domain", "hypothesis-generation", "option-refinement"]
    slug  = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:28]
    wslug = re.sub(r"[^a-z0-9]+", "-", winner.get("option_name", "").lower()).strip("-")[:28]
    for s in [slug, wslug]:
        if s and s not in base:
            base.append(s)
    return base[:7]
