"""Compute experiment step (4.5) for the sunnyday agent."""

import re

from hackathon_science.tools import run_code

from shared import MODEL_ID, call_llm, _extract_json, _text


def _generate_and_run_compute(topic: str, domain_brief: str, options: list[dict]) -> dict:
    """Ask LLM to write a Python experiment script, run it, return per-option metrics."""
    opt_lines = "\n".join(
        f"- {o.get('option_name','')}: hypothesis={o.get('hypothesis','')[:120]}; "
        f"mechanism={o.get('mechanism','')[:120]}"
        for o in options
    )
    prompt = (
        f"Write a self-contained Python script that runs a controlled experiment for the research topic: '{topic}'.\n\n"
        f"Domain brief:\n{domain_brief[:500]}\n\n"
        f"Research options to compare (implement each as a distinct strategy):\n{opt_lines}\n\n"
        "Requirements:\n"
        "1. Generate synthetic data appropriate to the domain — no file I/O, no external packages\n"
        "2. Implement each option's mechanism as a named function\n"
        "3. Run 25 independent trials per option; measure:\n"
        "   - mean_quality: float in [0,1] representing solution quality\n"
        "   - mean_cost: float representing relative compute cost\n"
        "   - std_quality: standard deviation of quality across trials\n"
        "4. Use only stdlib: random, math, statistics, json, hashlib\n"
        "5. Use pow(x, n) instead of x**n — do NOT write ** anywhere\n"
        "6. Do NOT write Python comments (no # lines)\n"
        "7. End the script with exactly these three lines:\n"
        "   print('COMPUTE_START')\n"
        "   print(json.dumps({'results': [...list of dicts...]}))\n"
        "   print('COMPUTE_END')\n"
        "8. Each dict in 'results' must have keys: option_name, mean_quality, mean_cost, std_quality\n"
        "9. Total runtime must be under 25 seconds\n"
        "10. Return ONLY the Python code — no markdown fences, no explanation"
    )
    try:
        response = call_llm(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            model_id=MODEL_ID,
            max_retries=2,
        )
        script = _text(response).strip()
        if not script:
            return {}
        script = _sanitise_script(script)
        output = run_code(script, filename="compute_experiment.py", timeout=60)
        payload = _extract_json(output, "COMPUTE_START", "COMPUTE_END")
        if isinstance(payload, dict) and isinstance(payload.get("results"), list) and payload["results"]:
            return payload
    except Exception:
        pass
    return {}


def _sanitise_script(code: str) -> str:
    """Strip markdown fences and replace ** to prevent extract_code_from_llm_response mangling."""
    code = re.sub(r"^```[a-zA-Z]*\n?", "", code.strip())
    if code.endswith("```"):
        code = code[:-3].strip()
    code = re.sub(r"(\w+)\s*\*\*\s*(\w+)", lambda m: f"pow({m.group(1)}, {m.group(2)})", code)
    code = code.replace("**", " ")
    return code
