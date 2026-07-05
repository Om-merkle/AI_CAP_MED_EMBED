"""Headless end-to-end runner for the medical embedding pipeline.

Runs every stage in order and prints the before/after result. This is what the Kaggle
notebook calls (Path A) and what you use locally for a quick CPU demo.

Examples
--------
# Full run on a Kaggle GPU (NFCorpus medical IR):
    python run_pipeline.py --domain nfcorpus --base-model BAAI/bge-small-en-v1.5 --epochs 1

# Train on the bundled medical flashcards dataset:
    python run_pipeline.py --domain flashcards --sample-size 5000

# Use MedEmbed's own clinical triplets (mining skipped automatically):
    python run_pipeline.py --domain medembed --sample-size 10000

# Tiny, fast CPU demo (skip the heavy official MTEB task):
    python run_pipeline.py --domain flashcards --sample-size 50 --eval-queries 30 --no-mteb
"""

from __future__ import annotations

import argparse
import json

from core.config import settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end medical embedding fine-tuning pipeline")
    p.add_argument("--domain", default=settings.domain,
                   help="nfcorpus | flashcards | medembed")
    p.add_argument("--base-model", default=settings.base_model)
    p.add_argument("--epochs", type=int, default=settings.epochs)
    p.add_argument("--batch-size", type=int, default=settings.batch_size)
    p.add_argument("--sample-size", type=int, default=settings.sample_size,
                   help="limit number of training pairs (default: all)")
    p.add_argument("--eval-queries", type=int, default=settings.eval_queries)
    p.add_argument("--num-negatives", type=int, default=settings.num_negatives)
    p.add_argument("--no-mteb", action="store_true", help="skip the official MTEB task (faster)")
    p.add_argument("--llm-triplets", action="store_true",
                   help="use LLM-generated clinical triplets instead of hard-negative mining")
    p.add_argument("--benchmark", action="store_true",
                   help="also benchmark candidate models (incl. MedEmbed) on the triplets")
    p.add_argument("--run-label", default="",
                   help="short name for this run, shown in the leaderboard")
    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    settings.domain = args.domain
    settings.base_model = args.base_model
    settings.epochs = args.epochs
    settings.batch_size = args.batch_size
    settings.sample_size = args.sample_size
    settings.eval_queries = args.eval_queries
    settings.num_negatives = args.num_negatives
    settings.run_mteb = not args.no_mteb


def main() -> None:
    args = parse_args()
    apply_args(args)

    # Imported after settings are applied so each stage sees the final config.
    from core import (
        baseline, benchmark, compare, data_prep, evaluate, leaderboard,
        llm_triplet_gen, train, triplet_mining,
    )

    print(f"[device] {settings.device}  |  base_model={settings.base_model}  "
          f"domain={settings.domain}  mteb_task={settings.effective_mteb_task}")

    print("\n[1/6] Preparing medical data ...")
    print(json.dumps(data_prep.build_pairs(), indent=2))

    print("\n[2/6] Collecting triplets ...")
    triplet_info = llm_triplet_gen.generate() if args.llm_triplets else triplet_mining.mine()
    print(json.dumps(triplet_info, indent=2))
    usage = triplet_info.get("usage")
    if usage:
        print(f"LLM usage: {usage['input_tokens']:,} input / {usage['output_tokens']:,} output "
              f"tokens ≈ ${usage['estimated_cost_usd']} ({usage['model']})")

    if args.benchmark:
        print("\n[extra] Benchmarking candidate models on the domain triplets ...")
        print(benchmark.show(benchmark.evaluate_models()))

    print("\n[3/6] MTEB / IR baseline (base model) ...")
    print(json.dumps(baseline.run(), indent=2))

    print("\n[4/6] Fine-tuning ...")
    print(json.dumps(train.finetune(), indent=2))

    print("\n[5/6] Evaluating fine-tuned model ...")
    print(json.dumps(evaluate.run(), indent=2))

    print("\n[6/6] Comparison (before vs after) ...")
    result = compare.diff()
    print(json.dumps(result, indent=2))

    delta = result.get("headline_ir_ndcg@10_delta")
    verdict = "IMPROVED" if result.get("improved") else "NO IMPROVEMENT"
    print(f"\n=== DONE: IR nDCG@10 delta = {delta}  ->  {verdict} ===")

    # Log this run and print the ranked leaderboard of all runs so far.
    leaderboard.record(run_label=args.run_label, num_triplets=triplet_info.get("num_triplets"))
    print("\n=== LEADERBOARD (best first) ===")
    print(leaderboard.show())


if __name__ == "__main__":
    main()
