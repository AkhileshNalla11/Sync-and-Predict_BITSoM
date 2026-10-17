"""PyTorch multi-task fusion model.

    static EHR vector ──► MLP ───────────────┐
                                             ├─ concat ─► shared MLP ─┬─► hyper head (logit)
    6 h sequence (72 x 13) ─► GRU | 1D-CNN ──┘                        └─► hypo head  (logit)

With ``use_static=False`` the static branch is removed (sequence-only twin, same architecture
otherwise), which is what the ablation compares against to isolate the value of EHR data.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

TARGETS = ("hyper", "hypo")


class GRUEncoder(nn.Module):
    """Final hidden state of a GRU over the window."""

    def __init__(self, n_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return h[-1]


class CNNEncoder(nn.Module):
    """Dilated 1D convolutions; returns [global mean pool, features at the last time step].

    The last-step features matter because the most recent glucose dominates short-term risk.
    """

    def __init__(self, n_in: int, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=8, dilation=4), nn.ReLU(),
        )
        self.out_dim = 2 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))              # (B, C, T)
        return torch.cat([h.mean(-1), h[..., -1]], dim=1)


class FusionNet(nn.Module):
    def __init__(self, n_seq: int, n_static: int, cfg: dict[str, Any], use_static: bool = True):
        super().__init__()
        p = cfg["fusion"]
        self.use_static = use_static and n_static > 0
        if p["encoder"] == "gru":
            self.seq_encoder: nn.Module = GRUEncoder(n_seq, p["seq_hidden"], p["gru_layers"], p["dropout"])
        elif p["encoder"] == "cnn":
            self.seq_encoder = CNNEncoder(n_seq, p["cnn_channels"], p["dropout"])
        else:
            raise ValueError(f"unknown encoder {p['encoder']!r}")
        d = self.seq_encoder.out_dim
        if self.use_static:
            self.static_encoder = nn.Sequential(
                nn.Linear(n_static, p["static_hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]),
                nn.Linear(p["static_hidden"], p["static_out"]), nn.ReLU())
            d += p["static_out"]
        self.shared = nn.Sequential(nn.Linear(d, p["shared_hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]))
        self.heads = nn.ModuleDict({t: nn.Linear(p["shared_hidden"], 1) for t in TARGETS})

    def forward(self, seq: torch.Tensor, static: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z = self.seq_encoder(seq)
        if self.use_static:
            if static is None:
                raise ValueError("this model needs the static vector")
            z = torch.cat([z, self.static_encoder(static)], dim=1)
        h = self.shared(z)
        return {t: head(h).squeeze(-1) for t, head in self.heads.items()}


def focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float, pos_weight: float = 1.0) -> torch.Tensor:
    """Binary focal loss; gamma=0 and pos_weight=1 reduce it to plain BCE."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p_t = torch.exp(-bce)
    w = torch.where(y > 0.5, torch.as_tensor(pos_weight, dtype=logits.dtype), torch.ones_like(y))
    return (w * (1 - p_t) ** gamma * bce).mean()


class MultiTaskLoss(nn.Module):
    """Plain BCE for hyper (common) + imbalance-aware loss for hypo (rare)."""

    def __init__(self, cfg: dict[str, Any], hypo_prevalence: float):
        super().__init__()
        p = cfg["fusion"]
        self.kind = p["hypo_loss"]
        self.gamma = p["focal_gamma"]
        self.hypo_weight = p["hypo_loss_weight"]
        # sqrt of the class ratio: up-weights rare positives without fully inverting the prior
        self.pos_weight = float(((1 - hypo_prevalence) / max(hypo_prevalence, 1e-6)) ** 0.5)

    def forward(self, logits: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> torch.Tensor:
        hyper = F.binary_cross_entropy_with_logits(logits["hyper"], y["hyper"])
        if self.kind == "weighted_bce":
            hypo = F.binary_cross_entropy_with_logits(logits["hypo"], y["hypo"],
                                                      pos_weight=torch.tensor(self.pos_weight))
        elif self.kind == "focal":
            hypo = focal_loss(logits["hypo"], y["hypo"], self.gamma)
        elif self.kind == "bce":
            hypo = F.binary_cross_entropy_with_logits(logits["hypo"], y["hypo"])
        else:
            raise ValueError(f"unknown hypo_loss {self.kind!r}")
        return hyper + self.hypo_weight * hypo
