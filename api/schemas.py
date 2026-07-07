"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConfigUpdate(BaseModel):
    """Subset of settings the UI is allowed to change at runtime."""

    base_model: str | None = None
    base_models: str | None = None      # comma-separated open-source sweep (empty = single)
    baseline_models: str | None = None  # comma-separated closed-source (encode-only) baselines
    domain: str | None = None
    mteb_tasks: str | None = None
    sample_size: int | None = None
    eval_queries: int | None = None
    num_negatives: int | None = None
    epochs: int | None = None
    batch_size: int | None = None
    run_mteb: bool | None = None


class JobRef(BaseModel):
    job_id: str
    kind: str


class Job(BaseModel):
    id: str
    kind: str
    status: str
    result: Any | None = None
    error: str | None = None
