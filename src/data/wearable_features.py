"""Rolling-window construction and labelling for the wearable stream.

Stage 1 uses only the labelling helpers (to calibrate the simulator against the
target event rates); Stage 2 adds the full window/feature builder on top.

Label definitions (shared by every model and every metric):
  * prediction time  t = last sample of the 6 h history window
  * horizon          = the 2 h of samples strictly after t
  * hyper label      = any observed CGM value  > hyper_mgdl in the horizon
  * hypo label       = any observed CGM value  < hypo_mgdl  in the horizon
  * windows with too little observed CGM in history or horizon are dropped
Labels use the *observed* CGM (with sensor noise and dropouts), as a deployed
system would, not the simulator's hidden true glucose.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def window_sizes(cfg: dict[str, Any]) -> tuple[int, int, int]:
    """(history, horizon, stride) in samples."""
    dt = cfg["sampling"]["interval_minutes"]
    w = cfg["windows"]
    return (w["history_hours"] * 60 // dt, w["horizon_hours"] * 60 // dt, w["stride_minutes"] // dt)


def anchor_indices(n_steps: int, history: int, horizon: int, stride: int) -> np.ndarray:
    """Indices of prediction times t with a full history [t-history+1, t] and horizon [t+1, t+horizon]."""
    return np.arange(history - 1, n_steps - horizon, stride)


def _rolling_stats(x: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nanmax, nanmin and observed fraction over each length-`width` window along the last axis.

    Output index i covers x[..., i : i + width].
    """
    win = sliding_window_view(x, width, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN windows
        return np.nanmax(win, axis=-1), np.nanmin(win, axis=-1), np.mean(~np.isnan(win), axis=-1)


def is_nocturnal(ts: pd.DatetimeIndex | pd.Series, start: str, end: str) -> np.ndarray:
    """True when the clock time falls in [start, end), handling windows that wrap midnight."""
    ts = pd.DatetimeIndex(ts)
    minutes = ts.hour * 60 + ts.minute
    s = int(start[:2]) * 60 + int(start[3:])
    e = int(end[:2]) * 60 + int(end[3:])
    m = np.asarray(minutes)
    return (m >= s) | (m < e) if s > e else (m >= s) & (m < e)


def label_windows(cgm: np.ndarray, timestamps: pd.DatetimeIndex, cfg: dict[str, Any]) -> pd.DataFrame:
    """Label every valid prediction time for a (n_patients, n_steps) CGM matrix.

    Returns one row per (patient index, anchor index) with hyper/hypo labels,
    a nocturnal flag, and coverage diagnostics.
    """
    history, horizon, stride = window_sizes(cfg)
    thr = cfg["thresholds"]
    n_pat, n_steps = cgm.shape
    anchors = anchor_indices(n_steps, history, horizon, stride)

    # history window ending at t starts at t-history+1; horizon starts at t+1
    _, _, hist_cov = _rolling_stats(cgm, history)
    fut_max, fut_min, fut_cov = _rolling_stats(cgm, horizon)
    h_cov = hist_cov[:, anchors - history + 1]
    f_cov, f_max, f_min = (a[:, anchors + 1] for a in (fut_cov, fut_max, fut_min))

    noct = is_nocturnal(timestamps[anchors], thr["nocturnal_start"], thr["nocturnal_end"])
    out = pd.DataFrame({
        "patient_idx": np.repeat(np.arange(n_pat), len(anchors)),
        "anchor_idx": np.tile(anchors, n_pat),
        "history_coverage": h_cov.ravel(),
        "horizon_coverage": f_cov.ravel(),
        "hyper": (f_max > thr["hyper_mgdl"]).ravel(),
        "hypo": (f_min < thr["hypo_mgdl"]).ravel(),
        "nocturnal": np.tile(noct, n_pat),
    })
    w = cfg["windows"]
    keep = (out.history_coverage >= w["min_history_coverage"]) & (out.horizon_coverage >= w["min_horizon_coverage"])
    return out[keep].reset_index(drop=True)
