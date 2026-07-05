"""OPTIONAL - LLM-based synthetic triplet generation (the MedEmbed recipe).

This is an ALTERNATIVE to auto hard-negative mining, mirroring how the original
MedEmbed models built their training data: clinical documents are fed to an LLM
which invents

  * a realistic clinical query the document answers  -> the POSITIVE pair, and
  * a plausible-but-wrong medical passage            -> a HARD NEGATIVE.

The prompt is adapted from MedEmbed's data-generation pipeline (which used
LLaMA 70B over PMC clinical notes) but works with any OpenAI-compatible API.
It is completely optional: if OPENAI_API_KEY is not set, `generate()` is a no-op
that tells you so. Output goes to the same data/triplets.jsonl consumed by
training, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from core.config import settings

_SYSTEM = (
    "You are a highly skilled medical AI assistant specializing in analyzing "
    "clinical and biomedical text and generating training data for a medical "
    "text-embedding retrieval model. Your expertise includes understanding complex "
    "medical terminology, identifying key clinical information, and formulating "
    "diverse, clinically relevant retrieval queries. Always respond in valid JSON."
)

_PROMPT = """Given the following medical DOCUMENT, return a JSON object with exactly these keys:

  "query"         - a realistic query a clinician, researcher or patient would type
                    that this document answers. Vary the style across calls: sometimes
                    keyword-based (e.g. "interatrial septal mass symptoms"), sometimes
                    a natural-language question, sometimes about treatment/procedure/follow-up.
  "hard_negative" - a short medically plausible passage on a RELATED topic that looks
                    relevant but does NOT actually answer the query (e.g. a different
                    condition, drug, or patient population).

Maintain clinical accuracy. Return ONLY the JSON object.

DOCUMENT:
{doc}"""


def _sample_docs(max_docs: int) -> list[str]:
    """Use the positives from stage 1 as raw documents if none are provided."""
    if not settings.pairs_path.exists():
        return []
    docs = []
    for line in settings.pairs_path.read_text(encoding="utf-8").splitlines():
        docs.append(json.loads(line)["positive"])
        if len(docs) >= max_docs:
            break
    return docs


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """$ estimate from the per-1M-token prices in settings (override in .env)."""
    cost = (
        input_tokens * settings.openai_input_price_per_1m
        + output_tokens * settings.openai_output_price_per_1m
    ) / 1_000_000
    return round(cost, 6)


def _usage_blob(input_tokens: int, output_tokens: int, num_calls: int) -> dict[str, Any]:
    return {
        "model": settings.openai_model,
        "num_api_calls": num_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": estimate_cost(input_tokens, output_tokens),
        "input_price_per_1m_usd": settings.openai_input_price_per_1m,
        "output_price_per_1m_usd": settings.openai_output_price_per_1m,
    }


def generate(docs: list[str] | None = None, max_docs: int = 100) -> dict[str, Any]:
    """Generate synthetic medical triplets. Requires OPENAI_API_KEY; otherwise a no-op."""
    if not settings.openai_api_key:
        return {"skipped": True, "reason": "OPENAI_API_KEY not set", "num_triplets": 0}

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    docs = docs or _sample_docs(max_docs)
    if not docs:
        return {"skipped": True, "reason": "no documents available", "num_triplets": 0}

    triplets: list[dict[str, str]] = []
    input_tokens = output_tokens = num_calls = 0
    for doc in docs[:max_docs]:
        try:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _PROMPT.format(doc=doc[:2000])},
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
            )
            num_calls += 1
            if resp.usage:
                input_tokens += resp.usage.prompt_tokens or 0
                output_tokens += resp.usage.completion_tokens or 0
            payload = json.loads(resp.choices[0].message.content)
            query, neg = payload.get("query"), payload.get("hard_negative")
            if query and neg:
                triplets.append({"anchor": query, "positive": doc, "negative": neg})
        except Exception:
            continue  # skip any doc the model/API fails on

    with settings.triplets_path.open("w", encoding="utf-8") as f:
        for row in triplets:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    usage = _usage_blob(input_tokens, output_tokens, num_calls)
    (settings.results_dir / "llm_usage.json").write_text(
        json.dumps(usage, indent=2), encoding="utf-8"
    )

    return {
        "skipped": False,
        "model": settings.openai_model,
        "num_triplets": len(triplets),
        "triplets_path": str(settings.triplets_path),
        "usage": usage,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
