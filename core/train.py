"""Stage 4 - Fine-tuning with sentence-transformers.

Uses the modern training stack:
  SentenceTransformer + SentenceTransformerTrainer + SentenceTransformerTrainingArguments
with MultipleNegativesRankingLoss, which uses the negative for each triplet AND
treats every other in-batch document as an extra negative - a strong, standard choice
for retrieval fine-tuning (the same contrastive objective family MedEmbed trained
its BGE variants with).

Device is auto-detected (CUDA on Kaggle/Colab, CPU locally). fp16 is enabled
automatically on GPU.
"""

from __future__ import annotations

import json
from typing import Any

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

from core.config import settings
from core.evaluate import TRIPLETS_EVAL_PATH
from core.triplet_mining import load_triplets


def finetune() -> dict[str, Any]:
    """Fine-tune the base model on medical triplets. Returns a small summary."""
    triplets = load_triplets()  # columns: anchor, positive, negative

    # Hold out a few triplets for the post-training TripletEvaluator accuracy check.
    holdout = min(200, max(0, len(triplets) // 20))
    if holdout:
        split = triplets.train_test_split(test_size=holdout, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
        with TRIPLETS_EVAL_PATH.open("w", encoding="utf-8") as f:
            for row in eval_ds:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    else:
        train_ds = triplets

    model = SentenceTransformer(settings.base_model, device=settings.device)
    # Cap sequence length to keep training memory in check on a single T4. Each triplet
    # feeds 3 texts through the model, so activation memory scales with seq length.
    if model.max_seq_length and model.max_seq_length > 256:
        model.max_seq_length = 256
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(settings.finetuned_model_dir / "checkpoints"),
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        # Transformers v5 deprecated `warmup_ratio`; `warmup_steps` given a float < 1.0 is
        # now interpreted as the warmup ratio (settings.warmup_ratio stays a fraction, e.g. 0.1).
        warmup_steps=settings.warmup_ratio,
        fp16=settings.use_fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # avoid duplicate in-batch negatives
        logging_steps=50,
        save_strategy="no",       # we save the final model manually below
        report_to=[],             # no wandb/tensorboard by default
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
    )
    trainer.train()

    model.save_pretrained(str(settings.finetuned_model_dir))

    return {
        "base_model": settings.base_model,
        "device": settings.device,
        "fp16": settings.use_fp16,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "num_train_triplets": len(train_ds),
        "num_holdout_triplets": holdout,
        "output_dir": str(settings.finetuned_model_dir),
    }


if __name__ == "__main__":
    print(json.dumps(finetune(), indent=2))
