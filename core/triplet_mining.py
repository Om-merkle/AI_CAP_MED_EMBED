"""Stage 2 - Triplet collection via hard-negative mining.

Turns the (anchor, positive) pairs from stage 1 into (anchor, positive, negative)
triplets. "Hard" negatives are documents that the base model *thinks* are similar to
the query but are NOT the labelled positive - training against these is what actually
teaches the model the medical domain (the same idea MedEmbed used, minus the LLM).

Uses `sentence_transformers.util.mine_hard_negatives`, the library's built-in miner.

For the ``medembed`` domain the triplets already ship with the dataset (stage 1 wrote
them), so mining short-circuits and simply reports the native triplets.
"""

from __future__ import annotations

import json
from typing import Any

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

from core.config import settings
from core.data_prep import domain_kind, load_pairs_dataset


def mine() -> dict[str, Any]:
    """Mine hard negatives and write data/triplets.jsonl. Returns a small summary."""
    if domain_kind() == "triplets" and settings.triplets_path.exists():
        n = sum(1 for l in settings.triplets_path.read_text(encoding="utf-8").splitlines() if l.strip())
        return {
            "source": "native (dataset ships expert triplets; mining skipped)",
            "num_triplets": n,
            "triplets_path": str(settings.triplets_path),
        }

    pairs = load_pairs_dataset()  # columns: anchor, positive
    model = SentenceTransformer(settings.base_model, device=settings.device)

    # The mining corpus is the set of unique positives. `range_max` (how deep we search)
    # must stay below that size or torch.topk overflows - so scale it to the corpus.
    n_docs = len(set(pairs["positive"]))
    range_max = min(50, max(2, n_docs - 5))
    range_min = min(10, max(0, range_max - 1))
    num_negatives = max(1, min(settings.num_negatives, range_max - range_min))

    triplets = mine_hard_negatives(
        pairs,
        model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        num_negatives=num_negatives,
        range_min=range_min,       # skip the very top hits (likely unlabelled positives)
        range_max=range_max,       # how deep to search for hard negatives
        absolute_margin=0.0,       # negative must be at least this much less similar
        sampling_strategy="top",
        batch_size=settings.batch_size,
        output_format="triplet",   # -> columns: anchor, positive, negative
        use_faiss=False,           # torch backend; avoids a hard faiss dependency
    )

    with settings.triplets_path.open("w", encoding="utf-8") as f:
        for row in triplets:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    return {
        "source": "mined (hard negatives via the base model)",
        "num_triplets": len(triplets),
        "num_negatives_per_pair": num_negatives,
        "triplets_path": str(settings.triplets_path),
        "columns": list(triplets.column_names),
    }


def load_triplets():
    """Return the triplets as a Hugging Face Dataset (anchor, positive, negative)."""
    from datasets import Dataset

    rows = [json.loads(l) for l in settings.triplets_path.read_text(encoding="utf-8").splitlines()]
    return Dataset.from_list(rows)


if __name__ == "__main__":
    print(json.dumps(mine(), indent=2))
