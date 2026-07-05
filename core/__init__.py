"""Medical embedding fine-tuning pipeline - shared core package.

Stages (each module is independently runnable with `python -m core.<module>`):

    data_prep       -> (anchor, positive) pairs + eval set from a medical dataset
    triplet_mining  -> (anchor, positive, negative) via hard-negative mining
    llm_triplet_gen -> OPTIONAL synthetic clinical triplets via an LLM (MedEmbed recipe)
    benchmark       -> shortlist candidate models on the domain triplets
    baseline        -> MTEB/IR "before" numbers for the base model
    train           -> fine-tune with MultipleNegativesRankingLoss
    evaluate        -> MTEB/IR "after" numbers for the fine-tuned model
    compare         -> before/after table
    leaderboard     -> ranked CSV of every run
"""

from core.config import settings  # noqa: F401
