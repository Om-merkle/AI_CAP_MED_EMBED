"""Central configuration for the whole medical embedding pipeline.

Every tunable lives here so the API, the Streamlit UI, the CLI and the Kaggle
notebook all behave identically. Values can be overridden with environment
variables (or a .env file) using the exact field name in UPPER_CASE,
e.g. `SAMPLE_SIZE=200`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains this `core/` package.
ROOT = Path(__file__).resolve().parent.parent

# Each medical domain maps to its closest official MTEB retrieval task — the
# "primary" benchmark used for the headline before/after number.
DOMAIN_MTEB_TASKS: dict[str, str] = {
    "nfcorpus": "NFCorpus",                    # medical/nutrition IR (BeIR)
    "flashcards": "MedicalQARetrieval",        # medical QA
    "medembed": "MedicalQARetrieval",          # MedEmbed clinical triplets
}

# The default medical benchmark suite: baseline AND fine-tuned models are evaluated
# on ALL of these, so the leaderboard shows a per-benchmark comparison (same suite
# as the MedEmbed paper minus TRECCOVID, which is ~171k docs and opt-in).
MTEB_BENCHMARK_TASKS: list[str] = [
    "MedicalQARetrieval", "PublicHealthQA", "NFCorpus", "ArguAna",
]
MTEB_ALL_TASKS: list[str] = MTEB_BENCHMARK_TASKS + ["TRECCOVID"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Model / domain ----
    base_model: str = "BAAI/bge-small-en-v1.5"
    domain: str = "nfcorpus"                # nfcorpus | flashcards | medembed
    mteb_task: str = ""                     # primary task; empty = auto from the domain
    mteb_tasks: str = ""                    # comma-separated suite; empty = MTEB_BENCHMARK_TASKS

    # ---- Data sizing (keep small for CPU demos, raise on Kaggle GPU) ----
    sample_size: int | None = None          # of training pairs; None = use all
    eval_queries: int = 100                 # eval queries for the quick IR metric
    eval_corpus_size: int = 5000            # cap corpus size for the quick IR metric
    num_negatives: int = 3                  # hard negatives mined per (query, positive)

    # ---- Training hyper-parameters ----
    epochs: int = 3                         # what the original MedEmbed trained with
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1

    # ---- Optional MTEB run (the "real" domain baseline; slower) ----
    run_mteb: bool = True

    # ---- Model shortlist benchmark (core/benchmark.py) ----
    benchmark_models: str = (
        "sentence-transformers/all-MiniLM-L6-v2,"
        "BAAI/bge-small-en-v1.5,"
        "abhinand/MedEmbed-small-v0.1"
    )

    # ---- Optional LLM triplet generation ----
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-nano"
    # $ per 1M tokens, used to estimate the LLM triplet-generation cost shown to the
    # user. Defaults match nano-class pricing; override in .env for other models.
    openai_input_price_per_1m: float = 0.05
    openai_output_price_per_1m: float = 0.40

    # ---- Optional Hub push ----
    hf_token: str = ""

    # ---- Paths (created on demand) ----
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"
    results_dir: Path = ROOT / "results"
    mteb_dir: Path = ROOT / "mteb_results"

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def use_fp16(self) -> bool:
        # fp16 only helps on CUDA; bge models train fine with it.
        return self.device == "cuda"

    @property
    def effective_mteb_task(self) -> str:
        """The primary benchmark (headline before/after number)."""
        return self.mteb_task or DOMAIN_MTEB_TASKS.get(self.domain, "NFCorpus")

    @property
    def effective_mteb_tasks(self) -> list[str]:
        """The full benchmark suite evaluated for baseline and fine-tuned models."""
        if self.mteb_tasks:
            names = [t.strip() for t in self.mteb_tasks.split(",") if t.strip()]
            return MTEB_ALL_TASKS if names == ["all"] else names
        suite = list(MTEB_BENCHMARK_TASKS)
        primary = self.effective_mteb_task
        if primary not in suite:
            suite.insert(0, primary)
        return suite

    @property
    def finetuned_model_dir(self) -> Path:
        return self.models_dir / f"{self.domain}-{Path(self.base_model).name}-ft"

    @property
    def raw_dir(self) -> Path:
        """Bundled/downloaded raw datasets live here (see scripts/download_dataset.py)."""
        return self.data_dir / "raw"

    @property
    def flashcards_raw_path(self) -> Path:
        return self.raw_dir / "medical_flashcards.jsonl"

    @property
    def medembed_raw_path(self) -> Path:
        return self.raw_dir / "medembed_triplets.jsonl"

    @property
    def pairs_path(self) -> Path:
        return self.data_dir / "pairs.jsonl"

    @property
    def triplets_path(self) -> Path:
        return self.data_dir / "triplets.jsonl"

    @property
    def eval_path(self) -> Path:
        return self.data_dir / "eval.json"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.raw_dir, self.models_dir, self.results_dir, self.mteb_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
