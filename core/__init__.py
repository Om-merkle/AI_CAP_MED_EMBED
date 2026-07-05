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

import warnings

# Silence harmless library chatter that clutters notebook/demo output:
# mteb still calls a renamed sentence-transformers method (FutureWarning per task),
# and transformers v5 prints a verbose "LOAD REPORT" for every model load.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"mteb(\..*)?")
try:
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()
except Exception:
    pass

from core.config import settings  # noqa: F401, E402
