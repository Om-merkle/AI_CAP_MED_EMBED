"""Stage 5a - Evaluation helpers.

Two ways to measure a model, both reported:

  1. A fast, self-contained Information-Retrieval metric on our compact medical
     eval set (`InformationRetrievalEvaluator`). It runs at any scale (even a
     50-sample CPU demo) and is computed identically for the base and fine-tuned
     model, so the before/after comparison is always apples-to-apples.

  2. The OFFICIAL MTEB medical task (NFCorpus / MedicalQARetrieval, matched to
     the domain) - the "real" benchmark number, the same family of tasks the
     original MedEmbed models were evaluated on. Slower and heavier, so it is
     optional (settings.run_mteb) and failures are caught rather than crashing
     the pipeline.

`ir_evaluate` / `mteb_evaluate` are reused by baseline.py (base model) and by
`run()` here (fine-tuned model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator, TripletEvaluator

from core.config import settings
from core.data_prep import load_eval

TRIPLETS_EVAL_PATH = settings.data_dir / "triplets_eval.jsonl"


def _pick(metrics: dict[str, float], needle: str) -> float | None:
    """Find a metric value by a suffix like 'ndcg@10' regardless of the prefix."""
    for key, value in metrics.items():
        if key.lower().endswith(needle.lower()):
            return round(float(value), 4)
    return None


def ir_evaluate(model: SentenceTransformer) -> dict[str, Any]:
    """Fast Information-Retrieval metrics on the compact eval set."""
    blob = load_eval()
    relevant = {qid: set(cids) for qid, cids in blob["relevant_docs"].items()}

    evaluator = InformationRetrievalEvaluator(
        queries=blob["queries"],
        corpus=blob["corpus"],
        relevant_docs=relevant,
        ndcg_at_k=[10],
        mrr_at_k=[10],
        map_at_k=[100],
        precision_recall_at_k=[10],
        accuracy_at_k=[1, 10],
        show_progress_bar=False,
        name=settings.domain,
    )
    raw = evaluator(model)
    return {
        "ndcg@10": _pick(raw, "ndcg@10"),
        "mrr@10": _pick(raw, "mrr@10"),
        "map@100": _pick(raw, "map@100"),
        "recall@10": _pick(raw, "recall@10"),
        "accuracy@1": _pick(raw, "accuracy@1"),
    }


_PREFERRED_SUBSETS = ("default", "eng", "en", "eng-eng")


def _extract_mteb_metrics(results: Any) -> dict[str, float | None]:
    """Pull ndcg_at_10 + mrr_at_5 (0-1 scale) out of an MTEB result, preferring English."""
    r = results[0] if isinstance(results, (list, tuple)) and results else results
    scores = getattr(r, "scores", None)
    if scores is None and isinstance(r, dict):
        scores = r.get("scores", r)

    entries: list[dict[str, Any]] = []
    try:
        for split_entries in scores.values():
            entries.extend(split_entries if isinstance(split_entries, list) else [split_entries])
    except Exception:
        return {"ndcg@10": None, "mrr@5": None}

    def rank(entry: dict[str, Any]) -> int:
        subset = str(entry.get("hf_subset", "default"))
        return _PREFERRED_SUBSETS.index(subset) if subset in _PREFERRED_SUBSETS else len(_PREFERRED_SUBSETS)

    for entry in sorted((e for e in entries if isinstance(e, dict)), key=rank):
        if "ndcg_at_10" in entry:
            return {
                "ndcg@10": round(float(entry["ndcg_at_10"]), 4),
                "mrr@5": round(float(entry["mrr_at_5"]), 4) if "mrr_at_5" in entry else None,
            }
    return {"ndcg@10": None, "mrr@5": None}


def mteb_evaluate(model: SentenceTransformer) -> dict[str, Any]:
    """Run the medical MTEB benchmark suite (settings.effective_mteb_tasks).

    Returns {'task': <primary>, 'ndcg@10': <primary score>, 'tasks': {name: metrics}}.
    The top-level task/ndcg@10 keep the primary-benchmark contract for compare/leaderboard.
    """
    per_task: dict[str, Any] = {}
    for task_name in settings.effective_mteb_tasks:
        try:
            import mteb

            tasks = mteb.get_tasks(tasks=[task_name])
            results = mteb.MTEB(tasks=tasks).run(
                model,
                output_folder=str(settings.mteb_dir),
                verbosity=0,
                overwrite_results=True,
            )
            per_task[task_name] = _extract_mteb_metrics(results)
        except Exception as exc:  # one bad task must never break the pipeline
            per_task[task_name] = {"ndcg@10": None, "mrr@5": None, "error": str(exc)}

    primary = settings.effective_mteb_task
    return {
        "task": primary,
        "ndcg@10": per_task.get(primary, {}).get("ndcg@10"),
        "tasks": per_task,
    }


def _triplet_accuracy(model: SentenceTransformer) -> float | None:
    """% of held-out triplets where positive is closer to anchor than negative."""
    if not TRIPLETS_EVAL_PATH.exists():
        return None
    rows = [json.loads(l) for l in TRIPLETS_EVAL_PATH.read_text(encoding="utf-8").splitlines()]
    if not rows:
        return None
    evaluator = TripletEvaluator(
        anchors=[r["anchor"] for r in rows],
        positives=[r["positive"] for r in rows],
        negatives=[r["negative"] for r in rows],
        show_progress_bar=False,
        name="triplet",
    )
    raw = evaluator(model)
    return _pick(raw, "cosine_accuracy") or _pick(raw, "accuracy")


def run(model_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate the FINE-TUNED model and write results/finetuned.json."""
    model_path = str(model_path or settings.finetuned_model_dir)
    model = SentenceTransformer(model_path, device=settings.device)

    result: dict[str, Any] = {
        "model": model_path,
        "ir": ir_evaluate(model),
        "triplet_accuracy": _triplet_accuracy(model),
    }
    if settings.run_mteb:
        result["mteb"] = mteb_evaluate(model)

    out = settings.results_dir / "finetuned.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
