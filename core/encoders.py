"""Encoder factory - one interface for open-source and closed-source embedders.

Every stage that scores a model (benchmark.py, med_leaderboard.py) used to call
`SentenceTransformer(name)` directly. That only works for open-source models on the
HuggingFace Hub. `load_encoder(name)` keeps that behaviour for ordinary names but
also understands an `openai:` prefix, returning a thin `OpenAIEmbedder` that talks to
the OpenAI embeddings API through the SAME `.encode(...)` surface the rest of the code
already relies on.

Closed-source models are baselines ONLY: we encode with them to compare against, but
never fine-tune them (their weights are not ours to train). Fine-tuning stays on the
open-source `SentenceTransformer` path in core/train.py.

Naming convention
-----------------
    "BAAI/bge-small-en-v1.5"          -> SentenceTransformer (open source, trainable)
    "openai:text-embedding-3-small"   -> OpenAIEmbedder      (closed source, baseline)

`OpenAIEmbedder.encode` accepts (and ignores) the extra keyword arguments MTEB passes
(`task_name`, `prompt_type`, ...), so an OpenAI baseline can appear in the medical MTEB
leaderboard too - though encoding a full retrieval corpus over the API costs real money,
so it is opt-in there, not a default.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.config import settings

OPENAI_PREFIX = "openai:"


def is_api_model(name: str) -> bool:
    """True for closed-source, API-backed models (encode-only baselines)."""
    return name.startswith(OPENAI_PREFIX)


def openai_model_id(name: str) -> str:
    """Strip the `openai:` prefix to get the raw OpenAI model id."""
    return name[len(OPENAI_PREFIX):] if is_api_model(name) else name


class OpenAIEmbedder:
    """Minimal, sentence-transformers-compatible wrapper over the OpenAI embeddings API.

    Only the surface the pipeline actually uses is implemented: `encode` (returning a
    numpy array or a torch tensor), plus `encode_queries` / `encode_corpus` aliases for
    MTEB. Token usage is accumulated on `.input_tokens` so callers can estimate cost.
    """

    def __init__(self, model: str, api_key: str, batch_size: int = 256, max_chars: int = 8000):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key)
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.input_tokens = 0  # accumulated across every encode() call

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        # Empty strings are rejected by the API; substitute a single space.
        cleaned = [(t[: self.max_chars] or " ") for t in batch]
        resp = self._client.embeddings.create(model=self.model, input=cleaned)
        if resp.usage:
            self.input_tokens += resp.usage.prompt_tokens or resp.usage.total_tokens or 0
        return [d.embedding for d in resp.data]

    def encode(
        self,
        sentences: Any,
        batch_size: int | None = None,
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,  # noqa: ARG002 - accepted for API parity
        normalize_embeddings: bool = False,
        **kwargs: Any,  # swallow MTEB's task_name / prompt_type / etc.
    ):
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        step = batch_size or self.batch_size

        vectors: list[list[float]] = []
        for start in range(0, len(texts), step):
            vectors.extend(self._embed_batch(texts[start : start + step]))

        arr = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings and arr.size:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.clip(norms, 1e-12, None)
        if single:
            arr = arr[0]

        if convert_to_tensor:
            import torch

            return torch.from_numpy(arr)
        return arr

    # MTEB (older APIs) calls these; both just defer to encode().
    def encode_queries(self, queries: Any, **kwargs: Any):
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: Any, **kwargs: Any):
        # MTEB passes corpus as dicts {"title": ..., "text": ...} or plain strings.
        if corpus and isinstance(corpus[0], dict):
            corpus = [(c.get("title", "") + " " + c.get("text", "")).strip() for c in corpus]
        return self.encode(corpus, **kwargs)


def load_encoder(name: str, device: str | None = None):
    """Return an encoder for `name`.

    Open-source names load a trainable `SentenceTransformer`; `openai:` names return an
    encode-only `OpenAIEmbedder`. Raises if a closed-source model is requested without an
    `OPENAI_API_KEY` (callers that want graceful skipping should check `is_api_model` +
    `settings.openai_api_key` first).
    """
    if is_api_model(name):
        if not settings.openai_api_key:
            raise RuntimeError(
                f"{name} needs OPENAI_API_KEY (set it in .env). Closed-source models are "
                "encode-only baselines."
            )
        return OpenAIEmbedder(openai_model_id(name), settings.openai_api_key)

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device or settings.device)
