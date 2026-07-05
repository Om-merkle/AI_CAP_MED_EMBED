"""Stage 3 - MTEB medical domain baseline.

Measures the UNTOUCHED base model on the medical domain, so we have a "before"
number to beat. Uses the same two metrics as post-training evaluation (a fast IR
metric plus the optional official MTEB medical task) to keep the comparison honest.
"""

from __future__ import annotations

import json
from typing import Any

from sentence_transformers import SentenceTransformer

from core.config import settings
from core.evaluate import ir_evaluate, mteb_evaluate


def run() -> dict[str, Any]:
    """Evaluate the base model and write results/baseline.json."""
    model = SentenceTransformer(settings.base_model, device=settings.device)

    result: dict[str, Any] = {
        "model": settings.base_model,
        "ir": ir_evaluate(model),
    }
    if settings.run_mteb:
        result["mteb"] = mteb_evaluate(model)

    out = settings.results_dir / "baseline.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
