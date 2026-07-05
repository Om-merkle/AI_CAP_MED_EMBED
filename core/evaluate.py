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


def _extract_mteb_ndcg(results: Any) -> float | None:
    """Pull ndcg_at_10 out of an MTEB result across library versions."""
    r = results[0] if isinstance(results, (list, tuple)) and results else results
    if hasattr(r, "get_score"):
        try:
            return round(float(r.get_score()), 4)
        except Exception:
            pass
    scores = getattr(r, "scores", None)
    if scores is None and isinstance(r, dict):
        scores = r.get("scores", r)
    try:
        for split_entries in scores.values():
            entries = split_entries if isinstance(split_entries, list) else [split_entries]
            for entry in entries:
                if isinstance(entry, dict) and "ndcg_at_10" in entry:
                    return round(float(entry["ndcg_at_10"]), 4)
    except Exception:
        pass
    return None


def mteb_evaluate(model: SentenceTransformer) -> dict[str, Any]:
    """Run the official MTEB medical task. Returns {'ndcg@10': ...} or an 'error'."""
    task_name = settings.effective_mteb_task
    try:
        import mteb

        tasks = mteb.get_tasks(tasks=[task_name])
        results = mteb.MTEB(tasks=tasks).run(
            model,
            output_folder=str(settings.mteb_dir),
            verbosity=0,
            overwrite_results=True,
        )
        return {"task": task_name, "ndcg@10": _extract_mteb_ndcg(results)}
    except Exception as exc:  # never let the optional benchmark break the pipeline
        return {"task": task_name, "ndcg@10": None, "error": str(exc)}


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
