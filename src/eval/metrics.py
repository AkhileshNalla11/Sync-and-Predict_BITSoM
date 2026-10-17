"""Shared evaluation metrics, used identically by every model.

Stage 3 provides the discrimination metrics; Stage 5 adds lead time and calibration.

Conventions:
  * scores: higher = more likely to have the event; need not be probabilities
  * CIs: 95 % percentile intervals from a PATIENT-level bootstrap (windows from one patient are
    strongly correlated, so resampling rows would make the intervals far too narrow)
  * for rare targets (hypoglycaemia) AUPRC is the headline metric; its chance level is the prevalence
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

WINDOWS_PER_DAY_DEFAULT = 48          # 30-min stride


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() > 0 else np.nan


def threshold_at_far(y: np.ndarray, s: np.ndarray, far: float) -> float:
    """Smallest score threshold whose false-positive rate (alert on score > thr) is <= `far`."""
    neg = np.sort(s[~y.astype(bool)])
    if len(neg) == 0:
        return np.inf
    k = int(np.floor(far * len(neg)))                 # at most k negatives may exceed the threshold
    return float(neg[len(neg) - k - 1]) if k < len(neg) else -np.inf


def sensitivity_at_far(y: np.ndarray, s: np.ndarray, far: float) -> float:
    if y.sum() == 0:
        return np.nan
    return float((s[y.astype(bool)] > threshold_at_far(y, s, far)).mean())


def patient_bootstrap(y: np.ndarray, s: np.ndarray, groups: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float],
                      n_boot: int, seed: int) -> tuple[float, float]:
    """95 % CI of fn(y, s) under resampling of whole patients with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[g] for g in rng.integers(0, len(uniq), len(uniq))])
        v = fn(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def evaluate_scores(y: np.ndarray, s: np.ndarray, groups: np.ndarray, cfg: dict[str, Any],
                    patient_days: float | None = None) -> dict[str, float]:
    """All discrimination metrics for one (target, subset, model).

    `patient_days` is the monitored time behind these windows (defaults to len(y) / 48, i.e. assumes
    a 30-min stride with no windows removed). Alert burden counts alerting windows, not episodes.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    ev = cfg["eval"]
    out: dict[str, float] = {"n_windows": int(len(y)), "n_patients": int(len(np.unique(groups))),
                             "n_positive": int(y.sum()), "prevalence": float(y.mean())}
    for name, fn in (("auroc", auroc), ("auprc", auprc)):
        out[name] = fn(y, s)
        out[f"{name}_lo"], out[f"{name}_hi"] = patient_bootstrap(y, s, groups, fn, ev["n_bootstrap"], cfg["seed"])
    patient_days = patient_days if patient_days is not None else len(y) / WINDOWS_PER_DAY_DEFAULT
    for far in ev["false_alert_rates"]:
        tag = f"{int(round(far * 100))}"
        thr = threshold_at_far(y, s, far)
        alerts = s > thr
        out[f"sens_at_far{tag}"] = sensitivity_at_far(y, s, far)
        out[f"ppv_at_far{tag}"] = float(y[alerts].mean()) if alerts.any() else np.nan
        out[f"alert_windows_per_patient_day_far{tag}"] = float(alerts.sum() / max(patient_days, 1e-9))
    return out


def evaluation_subsets(windows: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(target, subset, mask) triples.

    "onset" excludes windows where the patient is already past the threshold at prediction time;
    those are trivially positive, so onset is the headline subset. "all" follows the literal label.
    """
    every = np.ones(len(windows), bool)
    noct = windows.nocturnal.to_numpy(bool)
    onset = {t: ~windows[f"in_event_{t}"].to_numpy(bool) for t in ("hyper", "hypo")}
    return [("hyper", "onset", onset["hyper"]), ("hyper", "all", every),
            ("hypo", "onset", onset["hypo"]), ("hypo", "all", every),
            ("hypo", "nocturnal_onset", noct & onset["hypo"]), ("hypo", "nocturnal", noct)]


def evaluate_targets(windows: pd.DataFrame, scores: dict[str, np.ndarray], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """One metrics row per evaluation subset, given per-target scores aligned with `windows`."""
    rows = []
    stride_h = cfg["windows"]["stride_minutes"] / 60
    for target, subset, m in evaluation_subsets(windows):
        # monitored time = all windows of the patients in this subset (subsets drop windows, not time)
        pts = np.unique(windows.patient_id.to_numpy()[m])
        all_w = windows.patient_id.isin(pts).to_numpy()
        if subset.startswith("nocturnal"):
            all_w &= windows.nocturnal.to_numpy(bool)
        days = all_w.sum() * stride_h / 24
        res = evaluate_scores(windows[target].to_numpy()[m], scores[target][m], windows.patient_id.to_numpy()[m],
                              cfg, patient_days=days)
        rows.append({"target": target, "subset": subset, **res})
    return rows
