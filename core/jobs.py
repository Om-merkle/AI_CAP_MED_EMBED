"""A tiny in-process job registry.

Long stages (baseline, train, evaluate) run in a FastAPI BackgroundTask; the UI polls
`GET /jobs/{id}` for status. This is deliberately simple - a dict guarded by a lock -
because a single app process is all we need.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from typing import Any, Callable

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def create(kind: str) -> str:
    job_id = uuid.uuid4().hex[:8]
    with _LOCK:
        _JOBS[job_id] = {"id": job_id, "kind": kind, "status": "pending", "result": None, "error": None}
    return job_id


def get(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _set(job_id: str, **fields: Any) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(fields)


def run(job_id: str, fn: Callable[[], Any]) -> None:
    """Execute `fn`, recording status/result/error on the job. Meant for a background thread."""
    _set(job_id, status="running")
    try:
        result = fn()
        _set(job_id, status="done", result=result)
    except Exception as exc:  # surface the error to the UI instead of dying silently
        _set(job_id, status="error", error=f"{exc}\n{traceback.format_exc()}")
