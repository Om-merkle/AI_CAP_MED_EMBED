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

from core.config import settings

LEADERBOARD_PATH = settings.results_dir / "leaderboard.csv"

FIELDS = [
    "run_at", "run_label", "base_model", "domain", "device",
    "epochs", "batch_size", "sample_size", "num_triplets",
    "ir_ndcg@10_base", "ir_ndcg@10_ft", "ir_ndcg@10_delta",
    "mteb_ndcg@10_base", "mteb_ndcg@10_ft", "mteb_ndcg@10_delta",
    "triplet_accuracy",
]


def _read(name: str) -> dict[str, Any]:
    path = settings.results_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _delta(before: Any, after: Any) -> float | None:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return round(after - before, 4)
    return None


def record(run_label: str = "", num_triplets: int | None = None) -> dict[str, Any]:
    """Append the current run's metrics (from results/*.json) to the leaderboard CSV."""
    base, ft, cmp = _read("baseline.json"), _read("finetuned.json"), _read("comparison.json")
    mteb_b = base.get("mteb", {}).get("ndcg@10")
    mteb_f = ft.get("mteb", {}).get("ndcg@10")

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
        "mteb_ndcg@10_base": mteb_b,
        "mteb_ndcg@10_ft": mteb_f,
        "mteb_ndcg@10_delta": _delta(mteb_b, mteb_f),
        "triplet_accuracy": ft.get("triplet_accuracy"),
    }

    is_new = not LEADERBOARD_PATH.exists()
    with LEADERBOARD_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row


def load_rows() -> list[dict[str, Any]]:
    """Return all leaderboard rows, ranked best-first."""
    if not LEADERBOARD_PATH.exists():
        return []
    with LEADERBOARD_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def rank_key(r: dict[str, Any]) -> float:
        for col in ("mteb_ndcg@10_ft", "ir_ndcg@10_ft"):
            try:
                return float(r[col])
            except (TypeError, ValueError):
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

    cols = ["run_at", "run_label", "base_model", "domain", "epochs", "batch_size",
            "ir_ndcg@10_ft", "mteb_ndcg@10_ft", "mteb_ndcg@10_delta"]
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
