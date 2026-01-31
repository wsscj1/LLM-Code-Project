"""
ARR-LHRM: Adaptive Retrieval-Reasoning Hybrid Large Language Model.

This package provides:
- 3-mode inference orchestration: Direct / Retrieve / Reason
- A learnable router (policy network πϕ) for mode selection
- Training utilities for:
  - Stage I: Hybrid Fine-Tuning (HFT) cold start (router + optional LM formatting)
  - Stage II: HGPO-style reinforcement learning for the router (lightweight, runnable)
"""

