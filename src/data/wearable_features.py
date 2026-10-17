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


# --------------------------------------------------------------------------- series + windows
# Raw per-patient series, shape (n_patients, n_steps, len(SERIES_CHANNELS)); NaN = not observed.
SERIES_CHANNELS = ("cgm_mgdl", "steps", "heart_rate_bpm", "sleep_code")
SLEEP_CODES = {"awake": 0, "light": 1, "deep": 2, "rem": 3}

# Model-ready per-timestep features produced by encode_sequences.
SEQUENCE_FEATURES = ("cgm", "cgm_observed", "cgm_delta", "steps", "steps_observed", "hr", "hr_observed",
                     "asleep", "awake_in_bed", "deep", "rem", "tod_sin", "tod_cos")

# Fixed physiological scaling constants (not fitted, so they cannot leak test information).
CGM_CENTER, CGM_SCALE = 140.0, 50.0
HR_CENTER, HR_SCALE = 70.0, 15.0
STEPS_LOG_SCALE = float(np.log1p(600))


def to_series(linked: pd.DataFrame, patient_ids: list[str] | np.ndarray) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Pivot linked long-format wearable rows into a dense (P, T, C) float32 array in `patient_ids` order.

    All patients must share one regular timestamp grid (true for the simulated cohort).
    """
    timestamps = pd.DatetimeIndex(np.sort(linked.timestamp.unique()))
    df = linked.assign(sleep_code=linked.sleep_stage.map(SLEEP_CODES))
    df = df.set_index(["patient_id", "timestamp"])[list(SERIES_CHANNELS)]
    full = pd.MultiIndex.from_product([list(patient_ids), timestamps], names=["patient_id", "timestamp"])
    arr = df.reindex(full).to_numpy(dtype=np.float32)
    return arr.reshape(len(patient_ids), len(timestamps), len(SERIES_CHANNELS)), timestamps


def extract_windows(series: np.ndarray, patient_idx: np.ndarray, anchor_idx: np.ndarray, history: int) -> np.ndarray:
    """Raw history windows ending at (and including) each anchor: shape (N, history, C)."""
    offsets = np.arange(-history + 1, 1)
    return series[np.asarray(patient_idx)[:, None], np.asarray(anchor_idx)[:, None] + offsets]


def _ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along axis 1 of an (N, T) array; leading NaNs are back-filled."""
    n, t = x.shape
    obs = ~np.isnan(x)
    idx = np.maximum.accumulate(np.where(obs, np.arange(t), 0), axis=1)
    out = x[np.arange(n)[:, None], idx]
    first = np.where(obs.any(1), obs.argmax(1), 0)
    lead = np.arange(t)[None, :] < first[:, None]
    out[lead] = x[np.arange(n), first][:, None].repeat(t, 1)[lead]
    return out


def encode_sequences(raw: np.ndarray, anchor_idx: np.ndarray, start_hour: float, dt_min: int) -> np.ndarray:
    """Turn raw windows (N, H, C) into model-ready features (N, H, len(SEQUENCE_FEATURES)), NaN-free.

    Missing CGM / HR are forward-filled within the window and flagged by `*_observed`
    masks, so the model can tell a real reading from an imputed one.
    """
    n, h, _ = raw.shape
    cgm, steps, hr, sleep = (raw[..., i] for i in range(4))
    cgm_obs, steps_obs, hr_obs = (~np.isnan(a) for a in (cgm, steps, hr))

    cgm_f = np.nan_to_num(_ffill(cgm), nan=CGM_CENTER)
    hr_f = np.nan_to_num(_ffill(hr), nan=HR_CENTER)
    cgm_delta = np.diff(cgm_f, axis=1, prepend=cgm_f[:, :1]) / 10.0      # mg/dL per 5 min, scaled
    hours = (start_hour + (np.asarray(anchor_idx)[:, None] + np.arange(-h + 1, 1)) * dt_min / 60) % 24
    ang = 2 * np.pi * hours / 24

    feats = [
        (cgm_f - CGM_CENTER) / CGM_SCALE, cgm_obs, cgm_delta,
        np.log1p(np.nan_to_num(steps, nan=0.0)) / STEPS_LOG_SCALE, steps_obs,
        (hr_f - HR_CENTER) / HR_SCALE, hr_obs,
        np.isin(sleep, (1, 2, 3)), sleep == 0, sleep == 2, sleep == 3,
        np.sin(ang), np.cos(ang),
    ]
    return np.stack([np.asarray(f, dtype=np.float32) for f in feats], axis=-1)
