# Bundled medical datasets

Fetched by `python scripts/download_dataset.py` (re-run it any time to refresh or resize).

| File | Source | Rows | Format |
|---|---|---|---|
| `medical_flashcards.jsonl` | [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) | 33,955 (complete) | `{"input": <medical question>, "output": <answer>}` |
| `medembed_triplets.jsonl` | [abhinand/MedEmbed-training-triplets-v1](https://huggingface.co/datasets/abhinand/MedEmbed-training-triplets-v1) | 10,000 starter sample (complete = ~232k) | `{"query": ..., "pos": ..., "neg": ...}` |

The bundled triplets file is a **starter sample** so the repo stays small enough for GitHub.
`python scripts/download_dataset.py` (no flags) replaces it with the **complete ~232k-row
dataset** — the Colab and Kaggle notebooks run this automatically before training.

* `medical_flashcards.jsonl` powers the `flashcards` domain — hard negatives are mined.
* `medembed_triplets.jsonl` powers the `medembed` domain — these are the actual clinical
  triplets (with expert hard negatives) the published MedEmbed models were trained on,
  generated from PMC clinical notes via LLaMA 70B; mining is skipped.
* The default `nfcorpus` domain streams [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus)
  from the Hugging Face Hub on first use (it needs corpus/queries/qrels structure).
