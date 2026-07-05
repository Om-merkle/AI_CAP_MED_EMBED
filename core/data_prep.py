"""Stage 1 - Medical data preparation.

Supports three medical domains and builds the same two artifacts for each:

  1. (anchor, positive) training pairs                  -> data/pairs.jsonl
  2. a compact evaluation set (queries, corpus,
     relevant_docs) for a fast, consistent IR metric    -> data/eval.json

Domains
-------
* ``nfcorpus``   - BeIR/nfcorpus (medical & nutrition IR). Ships corpus / queries /
                   qrels and is an official MTEB retrieval task (``NFCorpus``), so
                   the baseline and post-training evaluation measure the same thing
                   on the same domain -> an honest before/after comparison.
* ``flashcards`` - medalpaca/medical_meadow_medical_flashcards. Medical Q/A pairs
                   (question = anchor, answer = positive). Loaded from the bundled
                   ``data/raw/medical_flashcards.jsonl`` when present (see
                   scripts/download_dataset.py), otherwise from the HF Hub.
* ``medembed``   - abhinand/MedEmbed-training-triplets-v1, the clinical triplets
                   behind the MedEmbed models (query / pos / neg). Because negatives
                   already exist, this domain ALSO writes data/triplets.jsonl
                   directly and the mining stage is skipped.
"""

from __future__ import annotations

import json
import random
from typing import Any

from datasets import Dataset, load_dataset

from core.config import settings

# domain key -> how to build training data
DOMAINS: dict[str, dict[str, str]] = {
    "nfcorpus": {"kind": "beir", "hf_id": "BeIR/nfcorpus"},
    "flashcards": {"kind": "qa_pairs", "hf_id": "medalpaca/medical_meadow_medical_flashcards"},
    "medembed": {"kind": "triplets", "hf_id": "abhinand/MedEmbed-training-triplets-v1"},
}

_RNG = random.Random(42)


def _load(hf_id: str, config: str | None = None, split: str | None = None):
    """load_dataset with a graceful fallback for datasets that need remote code."""
    try:
        return load_dataset(hf_id, config, split=split)
    except Exception:
        return load_dataset(hf_id, config, split=split, trust_remote_code=True)


def build_pairs() -> dict[str, Any]:
    """Build training pairs + an evaluation set for the configured medical domain."""
    spec = DOMAINS.get(settings.domain)
    if spec is None:
        raise ValueError(f"Unknown domain {settings.domain!r}. Choose from {sorted(DOMAINS)}")

    if spec["kind"] == "beir":
        return _build_from_beir(spec["hf_id"])
    if spec["kind"] == "qa_pairs":
        rows = _load_local_or_hub(settings.flashcards_raw_path, spec["hf_id"])
        # The medical question lives in `input`; `instruction` is a constant system prompt.
        pairs = _dedupe(
            {"anchor": (r.get("input") or "").strip(), "positive": (r.get("output") or "").strip()}
            for r in rows
        )
        return _finish_pairs(pairs, native_triplets=None)
    # kind == "triplets"
    rows = _load_local_or_hub(settings.medembed_raw_path, spec["hf_id"])
    triplets = _dedupe(
        {
            "anchor": (r.get("query") or "").strip(),
            "positive": (r.get("pos") or "").strip(),
            "negative": (r.get("neg") or "").strip(),
        }
        for r in rows
    )
    pairs = [{"anchor": t["anchor"], "positive": t["positive"]} for t in triplets]
    return _finish_pairs(pairs, native_triplets=triplets)


# ---- BeIR-style domains (corpus / queries / qrels) -----------------------------------

def _corpus_text(row: dict[str, Any]) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip()


def _build_lookup(hf_id: str) -> tuple[dict[str, str], dict[str, str]]:
    corpus_ds = _load(hf_id, "corpus")
    queries_ds = _load(hf_id, "queries")
    corpus_ds = corpus_ds["corpus"] if hasattr(corpus_ds, "keys") else corpus_ds
    queries_ds = queries_ds["queries"] if hasattr(queries_ds, "keys") else queries_ds

    corpus = {str(r["_id"]): _corpus_text(r) for r in corpus_ds}
    queries = {str(r["_id"]): (r["text"] or "").strip() for r in queries_ds}
    return corpus, queries


def _load_qrels(hf_id: str, split: str):
    qrels = _load(f"{hf_id}-qrels")
    if split not in qrels:
        split = "test" if "test" in qrels else next(iter(qrels.keys()))
    return qrels[split]


def _build_from_beir(hf_id: str) -> dict[str, Any]:
    corpus, queries = _build_lookup(hf_id)

    train_qrels = _load_qrels(hf_id, "train")
    pairs: list[dict[str, str]] = []
    for r in train_qrels:
        if int(r["score"]) <= 0:
            continue
        q = queries.get(str(r["query-id"]))
        d = corpus.get(str(r["corpus-id"]))
        if q and d:
            pairs.append({"anchor": q, "positive": d})

    _RNG.shuffle(pairs)
    if settings.sample_size:
        pairs = pairs[: settings.sample_size]
    _write_jsonl(settings.pairs_path, pairs)

    # ---- Evaluation set from the TEST qrels ----
    test_qrels = _load_qrels(hf_id, "test")
    relevant: dict[str, set[str]] = {}
    for r in test_qrels:
        if int(r["score"]) <= 0:
            continue
        qid, cid = str(r["query-id"]), str(r["corpus-id"])
        if qid in queries and cid in corpus:
            relevant.setdefault(qid, set()).add(cid)

    qids = list(relevant.keys())
    _RNG.shuffle(qids)
    qids = qids[: settings.eval_queries]
    eval_queries = {qid: queries[qid] for qid in qids}
    relevant = {qid: relevant[qid] for qid in qids}

    needed = set().union(*relevant.values()) if relevant else set()
    distractor_pool = [cid for cid in corpus.keys() if cid not in needed]
    _RNG.shuffle(distractor_pool)
    room = max(0, settings.eval_corpus_size - len(needed))
    corpus_ids = list(needed) + distractor_pool[:room]
    eval_corpus = {cid: corpus[cid] for cid in corpus_ids}

    blob = {
        "queries": eval_queries,
        "corpus": eval_corpus,
        "relevant_docs": {qid: sorted(cids) for qid, cids in relevant.items()},
    }
    settings.eval_path.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")

    return _summary(len(pairs), blob)


# ---- Pair/triplet-style domains (flashcards, medembed) --------------------------------

def _load_local_or_hub(local_path, hf_id: str) -> list[dict[str, Any]]:
    """Prefer the bundled JSONL in data/raw/ (offline, reproducible); else the HF Hub."""
    if local_path.exists():
        return [json.loads(l) for l in local_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ds = _load(hf_id, split="train")
    return [dict(r) for r in ds]


def _dedupe(rows) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for r in rows:
        if not r.get("anchor") or not r.get("positive"):
            continue
        key = r["anchor"]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _finish_pairs(pairs: list[dict[str, str]], native_triplets: list[dict[str, str]] | None) -> dict[str, Any]:
    """Shuffle, hold out an eval slice, apply sample_size, write all artifacts."""
    _RNG.shuffle(pairs)

    # Hold out eval queries FIRST so evaluation never sees training pairs.
    n_eval = min(settings.eval_queries, max(1, len(pairs) // 10))
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]
    if settings.sample_size:
        train_pairs = train_pairs[: settings.sample_size]

    _write_jsonl(settings.pairs_path, train_pairs)

    # Native triplets (medembed): keep only rows whose anchor is in the training split.
    if native_triplets is not None:
        train_anchors = {p["anchor"] for p in train_pairs}
        rows = [t for t in native_triplets if t["anchor"] in train_anchors]
        _write_jsonl(settings.triplets_path, rows)

    # Eval corpus = held-out positives + distractors drawn from the FULL dataset
    # (not just the sampled training pairs), so even tiny demos face a real corpus.
    eval_queries = {f"q{i}": p["anchor"] for i, p in enumerate(eval_pairs)}
    eval_corpus = {f"d{i}": p["positive"] for i, p in enumerate(eval_pairs)}
    relevant = {f"q{i}": [f"d{i}"] for i in range(len(eval_pairs))}

    room = max(0, settings.eval_corpus_size - len(eval_corpus))
    for j, p in enumerate(pairs[n_eval : n_eval + room]):
        eval_corpus[f"x{j}"] = p["positive"]

    blob = {"queries": eval_queries, "corpus": eval_corpus, "relevant_docs": relevant}
    settings.eval_path.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")

    return _summary(len(train_pairs), blob)


# ---- Shared helpers -------------------------------------------------------------------

def _write_jsonl(path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summary(num_pairs: int, eval_blob: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": settings.domain,
        "num_pairs": num_pairs,
        "num_eval_queries": len(eval_blob["queries"]),
        "eval_corpus_size": len(eval_blob["corpus"]),
        "pairs_path": str(settings.pairs_path),
        "eval_path": str(settings.eval_path),
    }


def domain_kind() -> str:
    """'beir' | 'qa_pairs' | 'triplets' for the configured domain."""
    return DOMAINS[settings.domain]["kind"]


# ---- Loaders used by later stages ------------------------------------------------------

def load_pairs_dataset() -> Dataset:
    """Return the (anchor, positive) pairs as a Hugging Face Dataset."""
    rows = [json.loads(line) for line in settings.pairs_path.read_text(encoding="utf-8").splitlines()]
    return Dataset.from_list(rows)


def load_eval() -> dict[str, Any]:
    """Return {queries, corpus, relevant_docs} for the Information-Retrieval metric."""
    return json.loads(settings.eval_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(json.dumps(build_pairs(), indent=2))
