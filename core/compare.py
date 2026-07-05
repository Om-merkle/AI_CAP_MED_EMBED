"""Stage 5b - Compare baseline vs fine-tuned.

Reads results/baseline.json and results/finetuned.json and produces a compact
before/after table (with deltas) -> results/comparison.json. The headline number is
the IR nDCG@10 improvement; the official MTEB medical-task nDCG@10 is included
when available.
"""

from __future__ import annotations

import json
from typing import Any

from core.config import settings


def _load(name: str) -> dict[str, Any]:
    path = settings.results_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the baseline / evaluate stages first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return round(after - before, 4)


def diff() -> dict[str, Any]:
    """Build the before/after comparison and write results/comparison.json."""
    base = _load("baseline.json")
    ft = _load("finetuned.json")

    rows = []
    for metric in ("ndcg@10", "mrr@10", "map@100", "recall@10"):
        b = base.get("ir", {}).get(metric)
        a = ft.get("ir", {}).get(metric)
        rows.append({"metric": f"IR {metric}", "baseline": b, "finetuned": a, "delta": _delta(b, a)})

    # One row per benchmark in the medical MTEB suite (4-5 tasks).
    b_tasks = base.get("mteb", {}).get("tasks", {})
    a_tasks = ft.get("mteb", {}).get("tasks", {})
    for task in dict.fromkeys(list(b_tasks) + list(a_tasks)):
        b = b_tasks.get(task, {}).get("ndcg@10")
        a = a_tasks.get(task, {}).get("ndcg@10")
        rows.append(
            {"metric": f"MTEB {task} ndcg@10", "baseline": b, "finetuned": a, "delta": _delta(b, a)}
        )

    # Back-compat: results produced before the multi-benchmark suite had a single score.
    if not b_tasks and not a_tasks:
        b_mteb = base.get("mteb", {}).get("ndcg@10")
        a_mteb = ft.get("mteb", {}).get("ndcg@10")
        if b_mteb is not None or a_mteb is not None:
            rows.append(
                {
                    "metric": f"MTEB {settings.effective_mteb_task} ndcg@10",
                    "baseline": b_mteb,
                    "finetuned": a_mteb,
                    "delta": _delta(b_mteb, a_mteb),
                }
            )

    headline = _delta(base.get("ir", {}).get("ndcg@10"), ft.get("ir", {}).get("ndcg@10"))
    result = {
        "base_model": base.get("model"),
        "finetuned_model": ft.get("model"),
        "headline_ir_ndcg@10_delta": headline,
        "improved": bool(headline and headline > 0),
        "triplet_accuracy_finetuned": ft.get("triplet_accuracy"),
        "rows": rows,
    }

    out = settings.results_dir / "comparison.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(diff(), indent=2))
