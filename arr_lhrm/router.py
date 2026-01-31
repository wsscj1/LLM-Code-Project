from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RouterConfig:
    base_model_name_or_path: str
    num_labels: int = 3
    max_length: int = 256


class RouterModel:
    """
    Lightweight policy network πϕ for selecting a mode among:
      direct / retrieve / reason

    Implementation: frozen transformer encoder + trainable linear head.
    """

    def __init__(self, cfg: RouterConfig):
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, use_fast=True, trust_remote_code=True)
        self.enc = AutoModel.from_pretrained(cfg.base_model_name_or_path, trust_remote_code=True)

        # Freeze encoder by default (fast + stable)
        for p in self.enc.parameters():
            p.requires_grad = False

        hidden = getattr(self.enc.config, "hidden_size", None)
        if hidden is None:
            # Some models use different config names
            hidden = getattr(self.enc.config, "d_model", None)
        if hidden is None:
            raise ValueError("Cannot infer encoder hidden size from config")

        self.head = nn.Linear(int(hidden), cfg.num_labels)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enc.to(self._device)
        self.head.to(self._device)

    @property
    def device(self):
        return self._device

    def parameters(self):
        # Only head is trainable by default.
        return self.head.parameters()

    def encode(self, queries: List[str]):
        import torch

        inputs = self.tok(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def logits(self, queries: List[str]):
        import torch

        inputs = self.encode(queries)
        out = self.enc(**inputs)
        # Use CLS if available; otherwise use first token hidden state.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            rep = out.pooler_output
        else:
            rep = out.last_hidden_state[:, 0, :]
        return self.head(rep)

    def probs(self, queries: List[str]):
        import torch

        lg = self.logits(queries)
        return torch.softmax(lg, dim=-1)

    def predict(self, query: str) -> int:
        import torch

        p = self.probs([query])[0]
        return int(torch.argmax(p).item())

    def save(self, out_dir: str) -> None:
        import os
        import json
        import torch

        os.makedirs(out_dir, exist_ok=True)
        # Save head weights and config; encoder is referenced by name.
        torch.save(self.head.state_dict(), os.path.join(out_dir, "router_head.pt"))
        with open(os.path.join(out_dir, "router_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                dict(
                    base_model_name_or_path=self.cfg.base_model_name_or_path,
                    num_labels=self.cfg.num_labels,
                    max_length=self.cfg.max_length,
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def load(dir_path: str) -> "RouterModel":
        import os
        import json
        import torch

        cfg_path = os.path.join(dir_path, "router_config.json")
        head_path = os.path.join(dir_path, "router_head.pt")
        cfg_dict = json.load(open(cfg_path, encoding="utf-8"))
        model = RouterModel(RouterConfig(**cfg_dict))
        sd = torch.load(head_path, map_location=model.device)
        model.head.load_state_dict(sd)
        return model

