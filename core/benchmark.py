"""Model shortlisting - domain-specific triplet benchmark.

Before committing to a base model, compare several candidates on the SAME medical
(query, positive, negative) triplets (adapted from the MED_EMBED_FT notebook).

Two numbers per model:
  * triplet accuracy - % of triplets where sim(query, positive) > sim(query, negative)
  * average margin   - mean of sim(query, positive) - sim(query, negative);
                       larger = a cleaner separation between right and wrong answers

Run AFTER stage 2 (data/triplets.jsonl must exist). Writes results/benchmark.json.
Candidates default to a general model, the BGE base and the published MedEmbed
model, so you can see exactly where your own fine-tune should land.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sentence_transformers import util

from core.config import settings
from core.encoders import is_api_model, load_encoder


def _candidate_models() -> list[str]:
    """Open-source candidates + closed-source baselines, deduped in order.

    Closed-source (`openai:`) baselines are dropped when OPENAI_API_KEY is unset, so the
    benchmark still runs on the open-source models alone.
    """
    names = [m.strip() for m in settings.benchmark_models.split(",") if m.strip()]
    names += [m.strip() for m in settings.baseline_models.split(",") if m.strip()]
    out: list[str] = []
    for name in names:
        if name in out:
            continue
        if is_api_model(name) and not settings.openai_api_key:
            continue  # closed-source baseline needs an API key; skip quietly
        out.append(name)
    return out


def _load_triplets(limit: int | None = 500) -> list[dict[str, str]]:
    if not settings.triplets_path.exists():
        raise FileNotFoundError(
            f"Missing {settings.triplets_path}. Run data prep + triplet collection first."
        )
    rows = [json.loads(l) for l in settings.triplets_path.read_text(encoding="utf-8").splitlines()]
    return rows[:limit] if limit else rows


def evaluate_models(model_names: list[str] | None = None, limit: int | None = 500) -> dict[str, Any]:
    """Benchmark candidate models on the domain triplets. Writes results/benchmark.json."""
    model_names = model_names or _candidate_models()
    rows = _load_triplets(limit)

    queries = [r["anchor"] for r in rows]
    positives = [r["positive"] for r in rows]
    negatives = [r["negative"] for r in rows]

    results: list[dict[str, Any]] = []
    for name in model_names:
        api = is_api_model(name)
        try:
            model = load_encoder(name)
        except Exception as exc:  # missing key / unreachable model: skip, don't crash
            results.append({"model": name, "api": api, "error": str(exc)})
            continue

        q = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        p = model.encode(positives, convert_to_tensor=True, show_progress_bar=False)
        n = model.encode(negatives, convert_to_tensor=True, show_progress_bar=False)

        pos_sim = util.cos_sim(q, p).diagonal()
        neg_sim = util.cos_sim(q, n).diagonal()
        margins = (pos_sim - neg_sim).cpu().numpy()

        row: dict[str, Any] = {
            "model": name,
            "api": api,  # closed-source baseline (True) vs trainable open-source (False)
            "triplet_accuracy": round(float((pos_sim > neg_sim).float().mean().item()), 4),
            "avg_margin": round(float(np.mean(margins)), 4),
            "min_margin": round(float(np.min(margins)), 4),
            "max_margin": round(float(np.max(margins)), 4),
        }
        # Attribute API token cost for closed-source baselines (encoded 3x len(rows) texts).
        if api and getattr(model, "input_tokens", 0):
            row["input_tokens"] = model.input_tokens
            row["estimated_cost_usd"] = round(
                model.input_tokens * settings.openai_embedding_price_per_1m / 1_000_000, 6
            )
        results.append(row)

    results.sort(
        key=lambda r: (r.get("triplet_accuracy") or -1.0, r.get("avg_margin") or -1.0),
        reverse=True,
    )
    blob = {"domain": settings.domain, "num_triplets": len(rows), "models": results}

    out = settings.results_dir / "benchmark.json"
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return blob


def show(blob: dict[str, Any] | None = None) -> str:
    """Render the benchmark as a printable ranked table."""
    if blob is None:
        path = settings.results_dir / "benchmark.json"
        if not path.exists():
            return "(no benchmark yet - run core.benchmark.evaluate_models first)"
        blob = json.loads(path.read_text(encoding="utf-8"))

    header = ["rank", "model", "kind", "triplet_accuracy", "avg_margin"]
    lines = [header]
    for i, r in enumerate(blob["models"], 1):
        kind = "api" if r.get("api") else "local"
        acc = "ERR" if "error" in r else str(r.get("triplet_accuracy"))
        margin = "-" if "error" in r else str(r.get("avg_margin"))
        lines.append([str(i), r["model"], kind, acc, margin])
    widths = [max(len(line[c]) for line in lines) for c in range(len(header))]
    return "\n".join("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(line)) for line in lines)


if __name__ == "__main__":
    print(show(evaluate_models()))
