from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RewardConfig:
    # weights
    w_accuracy: float = 1.0
    w_relevance: float = 0.3
    w_cost: float = 0.1


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def substring_match(pred: str, gold: str) -> float:
    p = pred.strip().lower()
    g = gold.strip().lower()
    if not g:
        return 0.0
    return 1.0 if g in p else 0.0


def compute_reward(
    *,
    pred: str,
    gold: Optional[str],
    relevance: float,
    cost_tokens: int,
    cfg: RewardConfig,
) -> float:
    """
    A simple composite reward used for HGPO-style router RL.

    - accuracy: EM / substring vs gold (if provided)
    - relevance: external signal (e.g., internal retrieval confidence)
    - cost: penalize longer outputs
    """
    acc = 0.0
    if gold:
        acc = max(exact_match(pred, gold), substring_match(pred, gold))
    cost = float(cost_tokens)
    # normalize cost roughly; keep it mild
    cost_pen = cost / 200.0
    return cfg.w_accuracy * acc + cfg.w_relevance * float(relevance) - cfg.w_cost * cost_pen

