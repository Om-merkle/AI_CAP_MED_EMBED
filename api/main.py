"""FastAPI backend.

Thin HTTP layer over the `core` package. Each pipeline stage is one endpoint. The
slow stages (prepare / mine / benchmark / baseline / train / evaluate) launch a
background job and return a job id; the client polls GET /jobs/{id}. `/compare`,
`/leaderboard` and `/status` are instant (they just read the results/ folder).
"""

from __future__ import annotations

import json
from typing import Callable

from fastapi import BackgroundTasks, FastAPI, HTTPException

from core import (
    baseline, benchmark, compare, data_prep, evaluate, jobs, leaderboard,
    llm_triplet_gen, train, triplet_mining,
)
from core.config import settings
from api.schemas import ConfigUpdate, Job, JobRef

app = FastAPI(title="Medical Embedding Fine-Tuning API", version="1.0.0")


def _launch(kind: str, fn: Callable[[], object], bg: BackgroundTasks) -> JobRef:
    job_id = jobs.create(kind)
    bg.add_task(jobs.run, job_id, fn)
    return JobRef(job_id=job_id, kind=kind)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": settings.device}


@app.get("/config")
def get_config() -> dict[str, object]:
    return {
        "base_model": settings.base_model,
        "domain": settings.domain,
        "domains": sorted(data_prep.DOMAINS),
        "mteb_task": settings.effective_mteb_task,
        "sample_size": settings.sample_size,
        "eval_queries": settings.eval_queries,
        "num_negatives": settings.num_negatives,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "run_mteb": settings.run_mteb,
        "device": settings.device,
    }


@app.post("/config")
def set_config(update: ConfigUpdate) -> dict[str, object]:
    for field, value in update.model_dump(exclude_none=True).items():
        setattr(settings, field, value)
    return get_config()


@app.get("/status")
def status() -> dict[str, bool]:
    """Which artifacts exist - lets the UI enable/disable steps."""
    return {
        "pairs": settings.pairs_path.exists(),
        "triplets": settings.triplets_path.exists(),
        "benchmark": (settings.results_dir / "benchmark.json").exists(),
        "baseline": (settings.results_dir / "baseline.json").exists(),
        "finetuned_model": settings.finetuned_model_dir.exists(),
        "evaluation": (settings.results_dir / "finetuned.json").exists(),
        "comparison": (settings.results_dir / "comparison.json").exists(),
    }


@app.post("/prepare-data", response_model=JobRef)
def prepare_data(bg: BackgroundTasks) -> JobRef:
    return _launch("prepare-data", data_prep.build_pairs, bg)


@app.post("/mine-triplets", response_model=JobRef)
def mine_triplets(bg: BackgroundTasks) -> JobRef:
    return _launch("mine-triplets", triplet_mining.mine, bg)


@app.post("/generate-triplets-llm", response_model=JobRef)
def generate_triplets_llm(bg: BackgroundTasks) -> JobRef:
    """OPTIONAL alternative to mining: synthetic clinical triplets via an LLM."""
    return _launch("generate-triplets-llm", llm_triplet_gen.generate, bg)


@app.post("/benchmark-models", response_model=JobRef)
def benchmark_models(bg: BackgroundTasks) -> JobRef:
    """Shortlist candidate models on the domain triplets (incl. MedEmbed)."""
    return _launch("benchmark-models", benchmark.evaluate_models, bg)


@app.post("/baseline", response_model=JobRef)
def run_baseline(bg: BackgroundTasks) -> JobRef:
    return _launch("baseline", baseline.run, bg)


@app.post("/train", response_model=JobRef)
def run_train(bg: BackgroundTasks) -> JobRef:
    return _launch("train", train.finetune, bg)


@app.post("/evaluate", response_model=JobRef)
def run_evaluate(bg: BackgroundTasks) -> JobRef:
    return _launch("evaluate", evaluate.run, bg)


@app.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return Job(**job)


@app.get("/compare")
def get_compare() -> dict[str, object]:
    try:
        return compare.diff()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/llm-usage")
def get_llm_usage() -> dict[str, object]:
    """Token usage + estimated cost of the last LLM triplet-generation run."""
    path = settings.results_dir / "llm_usage.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no LLM triplet generation has run yet")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/leaderboard")
def get_leaderboard() -> dict[str, object]:
    """Ranked table of all runs logged to results/leaderboard.csv (best first)."""
    return {"rows": leaderboard.load_rows()}


@app.get("/results/{name}")
def get_result(name: str) -> dict[str, object]:
    """Return a raw results JSON file (baseline / finetuned / comparison / benchmark)."""
    path = settings.results_dir / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name}.json not found")
    return json.loads(path.read_text(encoding="utf-8"))
