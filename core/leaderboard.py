"""Per-run leaderboard.

Appends one row per pipeline run to results/leaderboard.csv and renders a ranked
table, so you can compare experiments over time (different base models, epochs,
batch sizes, medical domains, ...). Rows are ranked by the official MTEB nDCG@10
of the fine-tuned model (falling back to the fast IR nDCG@10 when MTEB was skipped).

The CSV is intentionally the store of record: it survives across runs and can be
downloaded from Kaggle's Output tab.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from typing import Any

from core.config import MTEB_ALL_TASKS, settings

LEADERBOARD_PATH = settings.results_dir / "leaderboard.csv"

# Per-benchmark columns for the full medical MTEB suite (blank when a task wasn't run).
_TASK_FIELDS = [
    f"mteb_{task}_{suffix}" for task in MTEB_ALL_TASKS for suffix in ("base", "ft", "delta")
]

FIELDS = [
    "run_at", "run_label", "base_model", "domain", "device",
    "epochs", "batch_size", "sample_size", "num_triplets",
    "ir_ndcg@10_base", "ir_ndcg@10_ft", "ir_ndcg@10_delta",
    "mteb_primary_task", "mteb_primary_ft", "mteb_primary_delta",
    *_TASK_FIELDS,
    "triplet_accuracy",
    "llm_model", "llm_input_tokens", "llm_output_tokens", "llm_cost_usd",
]


def _read(name: str) -> dict[str, Any]:
    path = settings.results_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _delta(before: Any, after: Any) -> float | None:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return round(after - before, 4)
    return None


def record(
    run_label: str = "",
    num_triplets: int | None = None,
    llm_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append the current run's metrics (from results/*.json) to the leaderboard CSV.

    `llm_usage` is the usage blob from llm_triplet_gen (tokens + estimated cost of
    gpt-5.4-nano calls); pass it only when this run actually generated LLM triplets,
    so the cost columns stay attributed to the right run.
    """
    base, ft, cmp = _read("baseline.json"), _read("finetuned.json"), _read("comparison.json")
    mteb_b = base.get("mteb", {}).get("ndcg@10")
    mteb_f = ft.get("mteb", {}).get("ndcg@10")
    llm_usage = llm_usage or {}

    # Per-benchmark scores from the medical MTEB suite.
    b_tasks = base.get("mteb", {}).get("tasks", {})
    f_tasks = ft.get("mteb", {}).get("tasks", {})
    task_cols: dict[str, Any] = {}
    for task in MTEB_ALL_TASKS:
        tb = b_tasks.get(task, {}).get("ndcg@10")
        tf = f_tasks.get(task, {}).get("ndcg@10")
        task_cols[f"mteb_{task}_base"] = tb
        task_cols[f"mteb_{task}_ft"] = tf
        task_cols[f"mteb_{task}_delta"] = _delta(tb, tf)

    row: dict[str, Any] = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_label": run_label,
        "base_model": settings.base_model,
        "domain": settings.domain,
        "device": settings.device,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "sample_size": settings.sample_size,
        "num_triplets": num_triplets,
        "ir_ndcg@10_base": base.get("ir", {}).get("ndcg@10"),
        "ir_ndcg@10_ft": ft.get("ir", {}).get("ndcg@10"),
        "ir_ndcg@10_delta": cmp.get("headline_ir_ndcg@10_delta"),
        "mteb_primary_task": ft.get("mteb", {}).get("task") or settings.effective_mteb_task,
        "mteb_primary_ft": mteb_f,
        "mteb_primary_delta": _delta(mteb_b, mteb_f),
        **task_cols,
        "triplet_accuracy": ft.get("triplet_accuracy"),
        "llm_model": llm_usage.get("model"),
        "llm_input_tokens": llm_usage.get("input_tokens"),
        "llm_output_tokens": llm_usage.get("output_tokens"),
        "llm_cost_usd": llm_usage.get("estimated_cost_usd"),
    }

    _migrate_csv_if_needed()
    is_new = not LEADERBOARD_PATH.exists()
    with LEADERBOARD_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row


def _migrate_csv_if_needed() -> None:
    """Rewrite an existing CSV whose header predates the current FIELDS.

    Old rows keep their values under matching column names; columns that no longer
    exist are dropped and new columns stay blank — prevents silent misalignment.
    """
    if not LEADERBOARD_PATH.exists():
        return
    with LEADERBOARD_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == FIELDS:
            return
        old_rows = list(reader)
    with LEADERBOARD_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in old_rows:
            writer.writerow({k: r.get(k) for k in FIELDS})


def load_rows() -> list[dict[str, Any]]:
    """Return all leaderboard rows, ranked best-first."""
    if not LEADERBOARD_PATH.exists():
        return []
    with LEADERBOARD_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def rank_key(r: dict[str, Any]) -> float:
        for col in ("mteb_primary_ft", "mteb_ndcg@10_ft", "ir_ndcg@10_ft"):
            try:
                return float(r[col])
            except (KeyError, TypeError, ValueError):
                continue
        return -1.0

    return sorted(rows, key=rank_key, reverse=True)


def show(top: int | None = None) -> str:
    """Return a printable, ranked leaderboard table."""
    rows = load_rows()
    if not rows:
        return "(leaderboard empty - run the pipeline first)"
    if top:
        rows = rows[:top]

    # One nDCG@10 column per benchmark (fine-tuned score), TRECCOVID only when present.
    task_cols = [f"mteb_{t}_ft" for t in MTEB_ALL_TASKS
                 if t != "TRECCOVID" or any(r.get("mteb_TRECCOVID_ft") for r in rows)]
    cols = ["run_at", "run_label", "base_model", "domain", "epochs",
            "ir_ndcg@10_ft", *task_cols, "llm_cost_usd"]
    header = ["rank"] + cols
    lines = [header]
    for i, r in enumerate(rows, 1):
        lines.append([str(i)] + [str(r.get(c, "") or "-") for c in cols])

    widths = [max(len(line[c]) for line in lines) for c in range(len(header))]
    return "\n".join("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(line)) for line in lines)


def to_dataframe():
    """Return the ranked leaderboard as a pandas DataFrame (for notebook display)."""
    import pandas as pd

    rows = load_rows()
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=FIELDS)


if __name__ == "__main__":
    print(show())
