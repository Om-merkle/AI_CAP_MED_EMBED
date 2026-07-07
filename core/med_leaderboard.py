"""MedEmbed-style medical benchmark leaderboard.

Reproduces the "Medical / Clinical related Retrieval Benchmarks" table from the
MedEmbed project: every model is evaluated on the same set of official medical
MTEB retrieval tasks, reporting nDCG@10 and MRR@5 per task (as percentages),
plus the parameter count. The best score in each column is bolded and your own
fine-tuned model rows are highlighted, so you can see exactly where your model
lands against the published baselines.

Results are cached per (model, task) in results/med_benchmarks.json, so re-runs
only evaluate what is missing — add a model or a task and call evaluate() again.

Usage (notebook):
    from core import med_leaderboard
    med_leaderboard.evaluate(models=[...], tasks=[...])   # slow part, cached
    med_leaderboard.styled()                               # rich table (Colab/Kaggle)

Usage (CLI):
    python -m core.med_leaderboard --models BAAI/bge-small-en-v1.5,abhinand/MedEmbed-small-v0.1
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from core.config import settings

# The five tasks from the MedEmbed benchmark table. TRECCOVID is by far the
# heaviest (~171k documents) - drop it for a quick first pass.
DEFAULT_TASKS = ["TRECCOVID", "MedicalQARetrieval", "PublicHealthQA", "NFCorpus", "ArguAna"]
QUICK_TASKS = ["MedicalQARetrieval", "PublicHealthQA", "NFCorpus", "ArguAna"]

CACHE_PATH = settings.results_dir / "med_benchmarks.json"

_PREFERRED_SUBSETS = ("default", "english", "eng", "en", "eng-eng")


def _get_english_tasks(task_name: str):
    """Fetch a task restricted to English (PublicHealthQA etc. are multilingual)."""
    import mteb

    try:
        tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
        if tasks:
            return tasks
    except Exception:
        pass
    return mteb.get_tasks(tasks=[task_name])


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {"models": {}}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _count_params_millions(model) -> int | None:
    # Closed-source / API models have no local weights to count.
    if not hasattr(model, "_first_module"):
        return None
    n = sum(p.numel() for p in model._first_module().auto_model.parameters())
    return round(n / 1_000_000)


def _extract_metrics(results: Any) -> dict[str, float | None]:
    """Pull ndcg_at_10 and mrr_at_5 out of an MTEB result, preferring English subsets."""
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
                "ndcg@10": round(float(entry["ndcg_at_10"]) * 100, 2),
                "mrr@5": round(float(entry["mrr_at_5"]) * 100, 2) if "mrr_at_5" in entry else None,
            }
    return {"ndcg@10": None, "mrr@5": None}


def evaluate(models: list[str] | None = None, tasks: list[str] | None = None) -> dict[str, Any]:
    """Evaluate each model on each task (skipping cached pairs). Returns the cache."""
    import mteb

    from core.encoders import is_api_model, load_encoder

    models = models or [m.strip() for m in settings.benchmark_models.split(",") if m.strip()]
    tasks = tasks or DEFAULT_TASKS
    cache = _load_cache()

    for name in models:
        # API models have no param count, so `params_m is None` is not a "needs (re)load"
        # signal for them - track completeness by whether every task is cached instead.
        entry = cache["models"].setdefault(name, {"params_m": None, "tasks": {}})
        missing = [t for t in tasks if t not in entry["tasks"]]
        params_done = entry["params_m"] is not None or is_api_model(name)
        if not missing and params_done:
            print(f"[cached] {name}")
            continue

        print(f"[evaluating] {name} on {missing or '(params only)'}")
        try:
            model = load_encoder(name)
        except Exception as exc:  # e.g. closed-source model without an API key
            print(f"[skipped] {name}: {exc}")
            continue
        if entry["params_m"] is None:
            entry["params_m"] = _count_params_millions(model)

        for task_name in missing:
            try:
                task_objs = _get_english_tasks(task_name)
                results = mteb.MTEB(tasks=task_objs).run(
                    model,
                    output_folder=str(settings.mteb_dir),
                    verbosity=0,
                    overwrite_results=True,
                )
                entry["tasks"][task_name] = _extract_metrics(results)
            except Exception as exc:  # keep going; a failed task shows as blank
                entry["tasks"][task_name] = {"ndcg@10": None, "mrr@5": None, "error": str(exc)}
            _save_cache(cache)

    _save_cache(cache)
    return cache


def to_dataframe(tasks: list[str] | None = None):
    """Cached results as a DataFrame with (task, metric) MultiIndex columns."""
    import pandas as pd

    cache = _load_cache()
    tasks = tasks or DEFAULT_TASKS

    from core.encoders import is_api_model

    rows = []
    for name, entry in cache["models"].items():
        params = "API" if is_api_model(name) else f"{entry.get('params_m') or '?'}M"
        row: dict[Any, Any] = {("", "Model"): name, ("", "# Params"): params}
        for t in tasks:
            m = entry.get("tasks", {}).get(t, {})
            row[(t, "nDCG@10")] = m.get("ndcg@10")
            row[(t, "MRR@5")] = m.get("mrr@5")
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # Rank by average nDCG@10 across available tasks, best first.
    ndcg_cols = [c for c in df.columns if c[1] == "nDCG@10"]
    df = df.sort_values(by=ndcg_cols, key=lambda s: s.fillna(-1), ascending=False, ignore_index=True) if ndcg_cols else df
    return df


def styled(tasks: list[str] | None = None, highlight: list[str] | None = None):
    """MedEmbed-style rich table (for notebooks): bold best per column, highlight your models."""
    df = to_dataframe(tasks)
    if df.empty:
        print("(no benchmark results yet - run med_leaderboard.evaluate() first)")
        return df

    highlight = highlight or ["-ft"]  # fine-tuned models are highlighted by default
    model_col = ("", "Model")
    num_cols = [c for c in df.columns if c not in (model_col, ("", "# Params"))]

    def bold_max(col):
        is_max = col == col.max()
        return ["font-weight: bold" if v else "" for v in is_max]

    def highlight_rows(row):
        is_mine = any(h in str(row[model_col]) for h in highlight)
        style = "background-color: #d8f0d8" if is_mine else ""
        return [style] * len(row)

    styler = (
        df.style
        .apply(highlight_rows, axis=1)
        .apply(bold_max, subset=num_cols)
        .format({c: "{:.2f}" for c in num_cols}, na_rep="—")
        .set_caption("Medical / Clinical related Retrieval Benchmarks")
        .set_table_styles([
            {"selector": "caption",
             "props": "caption-side: top; font-size: 1.25em; font-weight: bold; padding: 8px;"},
            {"selector": "th", "props": "text-align: center;"},
        ])
        .hide(axis="index")
    )
    return styler


def main() -> None:
    p = argparse.ArgumentParser(description="MedEmbed-style medical benchmark leaderboard")
    p.add_argument("--models", default="", help="comma-separated model names (default: settings.benchmark_models)")
    p.add_argument("--tasks", default="", help=f"comma-separated MTEB tasks (default: {','.join(DEFAULT_TASKS)})")
    p.add_argument("--quick", action="store_true", help=f"skip TRECCOVID (use {','.join(QUICK_TASKS)})")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or (QUICK_TASKS if args.quick else None)

    evaluate(models=models, tasks=tasks)
    df = to_dataframe(tasks)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
