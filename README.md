# 🩺 AI — Medical Embedding Fine-Tuning (MedEmbed-style)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Om-merkle/AI_CAP_MED_EMBED/blob/main/notebooks/run_on_colab.ipynb)
[![Open In Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/kernels/welcome?src=https%3A%2F%2Fgithub.com%2FOm-merkle%2FAI_CAP_MED_EMBED%2Fblob%2Fmain%2Fnotebooks%2Frun_on_kaggle.ipynb)

**Notebooks:** [notebooks/run_on_colab.ipynb](https://github.com/Om-merkle/AI_CAP_MED_EMBED/blob/main/notebooks/run_on_colab.ipynb) · [notebooks/run_on_kaggle.ipynb](https://github.com/Om-merkle/AI_CAP_MED_EMBED/blob/main/notebooks/run_on_kaggle.ipynb)

An end-to-end application that fine-tunes a text-embedding model on **medical data** and
proves it got better on the medical domain. It reproduces the MedEmbed recipe with a simple,
understandable pipeline:

1. **Medical dataset + triplet collection** — real medical datasets, auto hard-negative
   mining (or optional LLM clinical-triplet generation, the original MedEmbed approach).
2. **MTEB medical baseline** — measure the base model on an official medical MTEB task
   *before* training.
3. **Fine-tune + evaluate + compare** — sentence-transformers fine-tuning, then an honest
   before/after comparison on the exact same metrics.

**Stack:** Python · sentence-transformers · MTEB · FastAPI · Streamlit
**Default model:** `BAAI/bge-small-en-v1.5` (the same base MedEmbed-small was tuned from)
**Default domain:** NFCorpus (medical IR, official MTEB task `NFCorpus`)



## How it works

One shared package (`core/`) holds all logic; the API, the UI, the CLI and the Kaggle
notebook all call the same functions.

```
prepare data → collect triplets → benchmark models → MTEB/IR baseline → fine-tune → evaluate → compare
 core/data_prep  core/triplet_mining  core/benchmark   core/baseline    core/train  core/evaluate  core/compare
```

| File | Role |
|---|---|
| `core/config.py` | all settings (model, domain, sizes, hyper-params) |
| `core/data_prep.py` | load a medical dataset → `(anchor, positive)` pairs + eval set |
| `core/triplet_mining.py` | `util.mine_hard_negatives` → `(anchor, positive, negative)` |
| `core/llm_triplet_gen.py` | *optional* synthetic clinical triplets via an LLM (MedEmbed recipe) |
| `core/benchmark.py` | shortlist candidate models (incl. MedEmbed) on the domain triplets |
| `core/baseline.py` | evaluate the base model (IR metric + official medical MTEB task) |
| `core/train.py` | `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss` |
| `core/evaluate.py` | evaluate the fine-tuned model |
| `core/compare.py` | before/after nDCG@10 table |
| `core/leaderboard.py` | ranked CSV of every run |
| `api/main.py` | FastAPI endpoints (long steps run as background jobs) |
| `app/streamlit_app.py` | click-through UI + before/after chart |
| `run_pipeline.py` | one-shot headless runner (CLI) |
| `scripts/download_dataset.py` | fetch the medical datasets into `data/raw/` |
| `notebooks/run_on_kaggle.ipynb` | run everything on a free Kaggle GPU |
| `notebooks/run_on_colab.ipynb` | run everything on a free Colab T4 GPU |

## Medical domains (datasets)

| `--domain` | Dataset | Where negatives come from | MTEB task |
|---|---|---|---|
| `nfcorpus` *(default)* | [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) — medical/nutrition IR with corpus, queries and qrels | hard-negative mining | `NFCorpus` |
| `flashcards` | [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) — all 33,955 medical Q/A pairs (bundled: `data/raw/medical_flashcards.jsonl`) | hard-negative mining | `MedicalQARetrieval` |
| `medembed` | [abhinand/MedEmbed-training-triplets-v1](https://huggingface.co/datasets/abhinand/MedEmbed-training-triplets-v1) — the clinical triplets the real MedEmbed models were trained on: **~232k rows complete** (a 10k starter sample is bundled at `data/raw/medembed_triplets.jsonl`) | **ships expert negatives** — mining is skipped | `MedicalQARetrieval` |

All data is **real** and pulled from the Hugging Face Hub. The Colab/Kaggle notebooks pull
the **complete** datasets automatically before training; locally:

```bash
python scripts/download_dataset.py                       # COMPLETE datasets (default: all ~232k triplets)
python scripts/download_dataset.py --medembed-rows 10000 # smaller local sample (keeps the repo light)
```

---

## ▶️ Run on Colab (free T4 GPU)

1. Push this project to GitHub.
2. Open `notebooks/run_on_colab.ipynb` in Colab (update the badge/repo URL to your username).
3. Runtime → **Change runtime type** → Hardware accelerator = **T4 GPU**.
4. Run the cells top-to-bottom — fully wired: GPU check → clone + install → *(optional)*
   `HF_TOKEN` from Colab secrets → **pull the complete datasets from HF** →
   `run_pipeline.py --epochs 3` → comparison + leaderboard → zip & download the model.

## ▶️ Run on Kaggle (alternative — free T4 ×2, 30 GPU-hrs/week)

1. Notebook file: **[notebooks/run_on_kaggle.ipynb](https://github.com/Om-merkle/AI_CAP_MED_EMBED/blob/main/notebooks/run_on_kaggle.ipynb)**
   — import it via Kaggle → **Create → Notebook** → **File → Import Notebook → GitHub** (paste that URL),
   or click the "Open in Kaggle" badge above (requires being logged in to Kaggle).
2. In **Settings**: **Accelerator = GPU T4 ×2**, **Internet = On**.
3. Run top-to-bottom — fully
   wired: GPU check → clone + install → *(optional)* `HF_TOKEN` from Kaggle secrets →
   **pull the complete datasets from HF** → **Path A**:
   `!python run_pipeline.py --domain nfcorpus --epochs 3 --batch-size 32 --benchmark`
   → print `results/comparison.json` → **fine-tuned nDCG@10 should beat the baseline**.
4. The model (`models/`) and metrics (`results/`) show up in the notebook's **Output** tab.
5. *(Optional)* **Path B** runs the FastAPI + Streamlit UI behind a public `cloudflared` URL.

## 💻 Run locally (Windows, CPU — tiny demo or UI)

```bash
pip install -r requirements.txt

# Tiny end-to-end demo on CPU (fast; skips the heavy official MTEB task):
python run_pipeline.py --domain flashcards --sample-size 50 --eval-queries 30 --no-mteb

# Or the full app (two terminals):
uvicorn api.main:app --reload            # terminal 1 → http://localhost:8000/docs
streamlit run app/streamlit_app.py       # terminal 2 → http://localhost:8501
```

> On CPU, keep `--sample-size` small. Full-scale training should run on Kaggle.

## Model shortlisting (MedEmbed-style benchmark)

Before fine-tuning you can compare candidate models on the *same* medical triplets —
triplet accuracy (`sim(q, pos) > sim(q, neg)`) and average margin:

```bash
python run_pipeline.py --domain medembed --sample-size 2000 --benchmark --no-mteb
# or standalone, after data prep + triplet collection:
python -m core.benchmark
```

Default candidates: `all-MiniLM-L6-v2` (general), `BAAI/bge-small-en-v1.5` (base) and
`abhinand/MedEmbed-small-v0.1` (published medical model) — so you can see exactly where
your own fine-tune should land.

## Optional: LLM clinical-triplet generation

This is how the original MedEmbed built its data (LLaMA 70B over PMC clinical notes).
Copy `.env.example` → `.env`, set `OPENAI_API_KEY`, then:

```bash
python run_pipeline.py --domain flashcards --llm-triplets --sample-size 50
```

If no key is set, this path is skipped automatically — mining remains the default.

**Token usage & cost are always shown**: every LLM run reports input tokens, output tokens
and the estimated cost per the configured model, e.g.
`LLM usage: 12,345 input / 6,789 output tokens ≈ $0.0034 (gpt-5.4-nano)`. The same numbers
persist in `results/llm_usage.json`, via `GET /llm-usage`, and as metric tiles in the
Streamlit UI. Prices are configurable in `.env`
(`OPENAI_INPUT_PRICE_PER_1M`, `OPENAI_OUTPUT_PRICE_PER_1M`).

## Multi-benchmark evaluation (4–5 medical MTEB tasks)

Every pipeline run evaluates the **base and fine-tuned models on a suite of 4 medical MTEB
benchmarks** — MedicalQARetrieval, PublicHealthQA, NFCorpus, ArguAna — and the comparison
plus the run leaderboard show per-benchmark before/after/delta columns. TRECCOVID (the 5th,
~171k docs) is opt-in:

```bash
python run_pipeline.py --domain nfcorpus --mteb-tasks all        # 5 benchmarks incl. TRECCOVID
python run_pipeline.py --mteb-tasks NFCorpus,ArguAna             # custom subset
python run_pipeline.py --no-mteb                                 # skip benchmarks (fast demo)
```

On a T4, the 4-task suite adds roughly 20–40 min (each benchmark runs twice: baseline and
fine-tuned). The headline metric remains the domain's primary task.

## What "success" looks like

`results/comparison.json` shows the base vs fine-tuned model side by side. Success = the
fine-tuned **IR nDCG@10** (and the official **medical MTEB nDCG@10**) is measurably higher
than the baseline — the same before/after story MedEmbed tells versus vanilla BGE.

## Medical benchmark leaderboard (MedEmbed-style)

The same "Medical / Clinical related Retrieval Benchmarks" table the MedEmbed project
publishes: every model evaluated on the official medical MTEB tasks (TRECCOVID,
MedicalQARetrieval, PublicHealthQA, NFCorpus, ArguAna), reporting **nDCG@10 + MRR@5 per
task** and **# params**, with the best score per column in bold and your fine-tuned models
highlighted. Both notebooks include a ready-made cell; standalone:

```bash
python -m core.med_leaderboard --quick        # skips TRECCOVID (the slow one)
python -m core.med_leaderboard --models BAAI/bge-small-en-v1.5,abhinand/MedEmbed-small-v0.1
```

```python
from core import med_leaderboard
med_leaderboard.evaluate(models=[...])   # cached per (model, task) in results/med_benchmarks.json
med_leaderboard.styled()                 # rich table in a notebook
```

Also in the Streamlit UI ("Medical benchmark leaderboard") and via the API
(`POST /med-leaderboard` to compute, `GET /med-leaderboard` for the cached results).

## Per-run leaderboard

Every run appends its metrics to `results/leaderboard.csv` and prints a ranked table
(best first). Runs that used LLM triplet generation also log their **token usage and
estimated cost** (`llm_model`, `llm_input_tokens`, `llm_output_tokens`, `llm_cost_usd`)
so the price of each experiment sits next to its scores. Tag a run with `--run-label`:

```bash
python run_pipeline.py --domain nfcorpus --run-label bge-nfcorpus-1ep
python run_pipeline.py --domain medembed --run-label bge-medembed-1ep
```

```python
from core import leaderboard
print(leaderboard.show())          # text table
leaderboard.to_dataframe()         # pandas DataFrame (nice in a notebook)
```

Or via the API: `GET /leaderboard`.
