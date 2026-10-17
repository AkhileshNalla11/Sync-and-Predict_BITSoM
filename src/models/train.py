"""Train the multi-task fusion model with early stopping on validation PATIENTS.

    python -m src.models.train                        # fusion GRU + sequence-only GRU (default)
    python -m src.models.train --variants fusion_cnn  # 1D-CNN encoder instead

Variants: fusion_gru, seqonly_gru, fusion_cnn, seqonly_cnn ("seqonly" drops the EHR branch).
Saves <models_dir>/<variant>.pt (weights) + <variant>.json (config, history, selection metric),
plus test-set metrics and predictions under <processed_dir>/results/.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.config import load_config, resolve_path
from src.data.build_dataset import load_processed
from src.data.ehr_features import STATIC_FEATURES
from src.data.wearable_features import SEQUENCE_FEATURES, encode_sequences, extract_windows, window_sizes
from src.eval.metrics import evaluate_targets
from src.models.baselines import results_markdown
from src.models.fusion_model import TARGETS, FusionNet, MultiTaskLoss


# --------------------------------------------------------------------------- data
class WindowSource:
    """Builds model-ready batches on the fly from the dense series (nothing per-window is stored)."""

    def __init__(self, ds: dict[str, Any], cfg: dict[str, Any]):
        self.series = ds["series"]
        self.history = window_sizes(cfg)[0]
        self.dt = cfg["sampling"]["interval_minutes"]
        ts0 = ds["timestamps"][0]
        self.start_hour = ts0.hour + ts0.minute / 60
        # static rows re-ordered to the series' patient order, so patient_idx indexes both
        self.static = ds["static"].loc[list(ds["patient_ids"])][list(STATIC_FEATURES)].to_numpy(np.float32)
        self.windows = ds["windows"]

    def batch(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        w = self.windows.iloc[rows]
        p_idx, a_idx = w.patient_idx.to_numpy(), w.anchor_idx.to_numpy()
        raw = extract_windows(self.series, p_idx, a_idx, self.history)
        seq = encode_sequences(raw, a_idx, self.start_hour, self.dt)
        y = {t: torch.from_numpy(w[t].to_numpy(np.float32)) for t in TARGETS}
        return torch.from_numpy(seq), torch.from_numpy(self.static[p_idx]), y


@torch.no_grad()
def predict(model: FusionNet, source: WindowSource, rows: np.ndarray, batch_size: int = 2048) -> dict[str, np.ndarray]:
    """Event probabilities for the given window rows."""
    model.eval()
    out: dict[str, list[np.ndarray]] = {t: [] for t in TARGETS}
    for a in range(0, len(rows), batch_size):
        seq, static, _ = source.batch(rows[a:a + batch_size])
        logits = model(seq, static)
        for t in TARGETS:
            out[t].append(torch.sigmoid(logits[t]).numpy())
    return {t: np.concatenate(v) if v else np.empty(0) for t, v in out.items()}


def onset_auprc(windows: pd.DataFrame, probs: dict[str, np.ndarray]) -> dict[str, float]:
    """Model-selection metric: AUPRC on windows not already in the event (matches the headline subset)."""
    res = {}
    for t in TARGETS:
        m = ~windows[f"in_event_{t}"].to_numpy(bool)
        y = windows[t].to_numpy(int)[m]
        res[t] = float(average_precision_score(y, probs[t][m])) if y.sum() else float("nan")
    return res


# --------------------------------------------------------------------------- training
def parse_variant(variant: str) -> tuple[bool, str]:
    kind, encoder = variant.split("_")
    if kind not in ("fusion", "seqonly") or encoder not in ("gru", "cnn"):
        raise ValueError(f"bad variant {variant!r}")
    return kind == "fusion", encoder


def train_model(ds: dict[str, Any], cfg: dict[str, Any], variant: str, log: bool = True) -> tuple[FusionNet, dict[str, Any]]:
    """Train on train patients; keep the epoch with the best validation onset AUPRC (mean of both targets)."""
    use_static, encoder = parse_variant(variant)
    cfg = copy.deepcopy(cfg)
    cfg["fusion"]["encoder"] = encoder
    p = cfg["fusion"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(p["num_threads"])
    torch.use_deterministic_algorithms(True)

    source = WindowSource(ds, cfg)
    split = ds["windows"].split.to_numpy()
    tr_rows, va_rows = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    val_w = ds["windows"].iloc[va_rows]

    model = FusionNet(len(SEQUENCE_FEATURES), len(STATIC_FEATURES), cfg, use_static=use_static)
    loss_fn = MultiTaskLoss(cfg, hypo_prevalence=float(ds["windows"].hypo.to_numpy()[tr_rows].mean()))
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

    best, best_state, best_epoch, bad, history = -np.inf, None, -1, 0, []
    for epoch in range(p["max_epochs"]):
        t0 = time.time()
        model.train()
        order = tr_rows[np.random.default_rng([cfg["seed"], 7, epoch]).permutation(len(tr_rows))]
        losses = []
        for a in range(0, len(order), p["batch_size"]):
            seq, static, y = source.batch(order[a:a + p["batch_size"]])
            loss = loss_fn(model(seq, static), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        val = onset_auprc(val_w, predict(model, source, va_rows))
        score = float(np.nanmean([val[t] for t in TARGETS]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auprc_hyper": val["hyper"],
                        "val_auprc_hypo": val["hypo"], "val_score": score, "seconds": round(time.time() - t0, 1)})
        if log:
            h = history[-1]
            print(f"  [{variant}] epoch {epoch:2d} loss {h['train_loss']:.4f} val AUPRC hyper {val['hyper']:.3f} "
                  f"hypo {val['hypo']:.3f} ({h['seconds']}s)", flush=True)
        if score > best + 1e-4:
            best, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= p["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    meta = {"variant": variant, "use_static": use_static, "encoder": encoder, "best_epoch": best_epoch,
            "best_val_score": best, "pos_weight_hypo": loss_fn.pos_weight, "fusion_config": p,
            "sequence_features": list(SEQUENCE_FEATURES), "static_features": list(STATIC_FEATURES),
            "history": history}
    return model, meta


def save_model(model: FusionNet, meta: dict[str, Any], models_dir: Path) -> None:
    torch.save(model.state_dict(), models_dir / f"{meta['variant']}.pt")
    (models_dir / f"{meta['variant']}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(variant: str, cfg: dict[str, Any]) -> tuple[FusionNet, dict[str, Any]]:
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    meta = json.loads((models_dir / f"{variant}.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["fusion"] = meta["fusion_config"]
    model = FusionNet(len(meta["sequence_features"]), len(meta["static_features"]), cfg, use_static=meta["use_static"])
    model.load_state_dict(torch.load(models_dir / f"{variant}.pt", weights_only=True))
    model.eval()
    return model, meta


# --------------------------------------------------------------------------- runner
def run(cfg: dict[str, Any], variants: list[str]) -> pd.DataFrame:
    ds = load_processed(cfg)
    source = WindowSource(ds, cfg)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    results_dir = resolve_path(Path(cfg["paths"]["processed_dir"]) / "results")
    te_rows = np.flatnonzero(ds["windows"].split.to_numpy() == "test")
    test_w = ds["windows"].iloc[te_rows].reset_index(drop=True)

    rows, preds = [], {"window_id": te_rows}
    for v in variants:
        model, meta = train_model(ds, cfg, v)
        save_model(model, meta, models_dir)
        probs = predict(model, source, te_rows)
        rows += [{"model": v, **r} for r in evaluate_targets(test_w, probs, cfg)]
        preds.update({f"{v}_{t}": probs[t] for t in TARGETS})
        print(f"  [{v}] best epoch {meta['best_epoch']}, val score {meta['best_val_score']:.3f}", flush=True)

    # Merge with results of variants trained in earlier runs (retrained variants are replaced).
    res = pd.DataFrame(rows)
    pred_df = pd.DataFrame(preds)
    metrics_path, preds_path = results_dir / "fusion_metrics.csv", results_dir / "fusion_test_predictions.csv.gz"
    # Only merge when the earlier run used the same test windows (i.e. the same data).
    if metrics_path.exists() and preds_path.exists():
        old_p = pd.read_csv(preds_path)
        if len(old_p) == len(pred_df) and (old_p.window_id.to_numpy() == pred_df.window_id.to_numpy()).all():
            keep = [c for c in old_p.columns if c != "window_id" and c not in pred_df.columns]
            pred_df = pd.concat([pred_df, old_p[keep]], axis=1)
            old = pd.read_csv(metrics_path)
            res = pd.concat([old[~old.model.isin(variants)], res], ignore_index=True)
    res.to_csv(metrics_path, index=False)
    pred_df.to_csv(preds_path, index=False)
    sub_order = {"onset": 0, "all": 1, "nocturnal_onset": 2, "nocturnal": 3}
    res = res.sort_values(["target", "subset"], key=lambda c: c.map(sub_order) if c.name == "subset" else c, kind="stable")
    (resolve_path(cfg["paths"]["docs_dir"]) / "fusion_results.md").write_text(
        results_markdown(res, "Stage 4 fusion model results"), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the multi-task fusion model")
    ap.add_argument("--config", default=None)
    ap.add_argument("--variants", default="fusion_gru,seqonly_gru")
    args = ap.parse_args()
    res = run(load_config(args.config), args.variants.split(","))
    cols = ["target", "subset", "model", "prevalence", "auprc", "auprc_lo", "auprc_hi", "auroc", "sens_at_far10"]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
