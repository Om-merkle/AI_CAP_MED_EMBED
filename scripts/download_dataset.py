"""Download the medical datasets into data/raw/ so the pipeline can run offline.

Fetches:
  1. medalpaca/medical_meadow_medical_flashcards -> data/raw/medical_flashcards.jsonl
     (all ~34k medical Q/A pairs; used by the `flashcards` domain)
  2. abhinand/MedEmbed-training-triplets-v1 -> data/raw/medembed_triplets.jsonl
     (the clinical (query, pos, neg) triplets behind the MedEmbed models;
      used by the `medembed` domain. COMPLETE dataset = ~232k rows / ~100MB JSONL)

Both files are plain JSONL, one record per line, so they are easy to inspect,
version and ship with the project.

Usage:
    python scripts/download_dataset.py                      # COMPLETE datasets (default)
    python scripts/download_dataset.py --medembed-rows 10000  # smaller local sample
    python scripts/download_dataset.py --medembed-rows 0    # skip the triplets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/download_dataset.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from core.config import settings  # noqa: E402


def _write_jsonl(path: Path, rows, keys: list[str]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k, "") for k in keys}, ensure_ascii=False) + "\n")
            n += 1
    return n


def download_flashcards() -> None:
    print("Downloading medalpaca/medical_meadow_medical_flashcards ...")
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    # `instruction` is a constant system prompt; the medical question is in `input`.
    n = _write_jsonl(settings.flashcards_raw_path, ds, ["input", "output"])
    print(f"  wrote {n} rows -> {settings.flashcards_raw_path}")


def download_medembed(max_rows: int | None) -> None:
    """max_rows=None -> the COMPLETE dataset (~232k triplets); 0 -> skip."""
    if max_rows is not None and max_rows <= 0:
        print("Skipping MedEmbed triplets (--medembed-rows 0)")
        return
    if max_rows is None:
        print("Downloading abhinand/MedEmbed-training-triplets-v1 (COMPLETE, ~232k rows) ...")
        ds = load_dataset("abhinand/MedEmbed-training-triplets-v1", split="train")
        rows = ds
    else:
        print(f"Downloading abhinand/MedEmbed-training-triplets-v1 (first {max_rows} rows, streamed) ...")
        stream = load_dataset("abhinand/MedEmbed-training-triplets-v1", split="train", streaming=True)
        rows = []
        for r in stream:
            rows.append(r)
            if len(rows) >= max_rows:
                break
    n = _write_jsonl(settings.medembed_raw_path, rows, ["query", "pos", "neg"])
    print(f"  wrote {n} rows -> {settings.medembed_raw_path}")


def _rows_arg(value: str) -> int | None:
    return None if value.lower() == "all" else int(value)


def main() -> None:
    p = argparse.ArgumentParser(description="Download the medical datasets into data/raw/")
    p.add_argument("--medembed-rows", type=_rows_arg, default=None,
                   help="how many MedEmbed triplets to fetch: 'all' (default, complete "
                        "dataset ~232k rows), an integer sample size, or 0 to skip")
    p.add_argument("--skip-flashcards", action="store_true")
    args = p.parse_args()

    settings.ensure_dirs()
    if not args.skip_flashcards:
        download_flashcards()
    download_medembed(args.medembed_rows)
    print("\nDone. The `flashcards` and `medembed` domains now run fully offline.")
    print("(`nfcorpus` streams BeIR/nfcorpus from the HF Hub on first use.)")


if __name__ == "__main__":
    main()
