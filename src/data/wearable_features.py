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
    # Current state at prediction time (last observed reading). Windows already past a threshold are
    # "ongoing" events; the clinically useful question is event ONSET, so metrics report both.
    current = _ffill(cgm.astype(np.float64))[:, anchors]
    out = pd.DataFrame({
        "patient_idx": np.repeat(np.arange(n_pat), len(anchors)),
        "anchor_idx": np.tile(anchors, n_pat),
        "history_coverage": h_cov.ravel(),
        "horizon_coverage": f_cov.ravel(),
        "hyper": (f_max > thr["hyper_mgdl"]).ravel(),
        "hypo": (f_min < thr["hypo_mgdl"]).ravel(),
        "nocturnal": np.tile(noct, n_pat),
        "in_event_hyper": (current > thr["hyper_mgdl"]).ravel(),
        "in_event_hypo": (current < thr["hypo_mgdl"]).ravel(),
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


# --------------------------------------------------------------------------- handcrafted features (Stage 3)
NOT_IN_WINDOW_MIN = 400.0      # "minutes since X" when X did not happen in the 6 h history

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "cgm": ("cgm_last", "cgm_minutes_since_obs", "cgm_mean", "cgm_std", "cgm_cv", "cgm_min", "cgm_max",
            "cgm_mean_1h", "cgm_min_1h", "cgm_max_1h", "cgm_slope_30m", "cgm_slope_60m", "cgm_accel",
            "cgm_delta_15m", "cgm_frac_above_180", "cgm_frac_below_70", "min_since_above_180",
            "min_since_below_70", "min_since_meal_rise", "max_rise_3h", "cgm_coverage"),
    "time": ("hour", "tod_sin", "tod_cos"),
    "wearables": ("steps_30m", "steps_1h", "steps_3h", "steps_max_1h", "min_since_active", "wear_coverage",
                  "hr_last", "hr_mean_30m", "hr_minus_6h_mean", "asleep_now", "asleep_minutes",
                  "awake_in_bed_minutes", "deep_frac"),
}


def _minutes_since(cond: np.ndarray, dt_min: int) -> np.ndarray:
    """Minutes from the last True in each row to the end of the window (NOT_IN_WINDOW_MIN if never)."""
    h = cond.shape[1]
    any_ = cond.any(1)
    last = h - 1 - np.argmax(cond[:, ::-1], axis=1)
    return np.where(any_, (h - 1 - last) * dt_min, NOT_IN_WINDOW_MIN).astype(np.float32)


def _slope(y: np.ndarray, dt_min: int) -> np.ndarray:
    """NaN-aware least-squares slope (mg/dL per minute) along axis 1; NaN if < 3 observed points."""
    x = np.arange(y.shape[1], dtype=np.float64) * dt_min
    w = ~np.isnan(y)
    n = w.sum(1)
    xm = np.where(w, x, 0).sum(1) / np.maximum(n, 1)
    ym = np.nansum(y, 1) / np.maximum(n, 1)
    dx = np.where(w, x[None, :] - xm[:, None], 0)
    dy = np.where(w, y - ym[:, None], 0)
    den = (dx ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (dx * dy).sum(1) / den
    return np.where(n >= 3, s, np.nan).astype(np.float32)


def handcrafted_features(raw: np.ndarray, anchor_idx: np.ndarray, start_hour: float, dt_min: int,
                         cfg: dict[str, Any]) -> pd.DataFrame:
    """Summary features for raw windows (N, H, C) -> DataFrame with every column in FEATURE_GROUPS.

    NaN is kept where a quantity is undefined (LightGBM handles missing values natively).
    """
    f = cfg["features"]
    sph = 60 // dt_min                                   # samples per hour
    cgm, steps, hr, sleep = (raw[..., i].astype(np.float64) for i in range(4))
    cgm_f = _ffill(cgm)
    obs = ~np.isnan(cgm)
    out: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out["cgm_last"] = cgm_f[:, -1]
        out["cgm_minutes_since_obs"] = _minutes_since(obs, dt_min)
        out["cgm_mean"] = np.nanmean(cgm, 1)
        out["cgm_std"] = np.nanstd(cgm, 1)
        out["cgm_cv"] = out["cgm_std"] / out["cgm_mean"]
        out["cgm_min"], out["cgm_max"] = np.nanmin(cgm, 1), np.nanmax(cgm, 1)
        last_h = cgm[:, -sph:]
        out["cgm_mean_1h"], out["cgm_min_1h"], out["cgm_max_1h"] = np.nanmean(last_h, 1), np.nanmin(last_h, 1), np.nanmax(last_h, 1)
        out["cgm_slope_30m"] = _slope(cgm[:, -sph // 2 - 1:], dt_min)
        out["cgm_slope_60m"] = _slope(cgm[:, -sph - 1:], dt_min)
        out["cgm_accel"] = out["cgm_slope_30m"] - _slope(cgm[:, -sph - 1: -sph // 2], dt_min)
        out["cgm_delta_15m"] = cgm_f[:, -1] - cgm_f[:, -4]
        n_obs = np.maximum(obs.sum(1), 1)
        out["cgm_frac_above_180"] = (cgm > 180).sum(1) / n_obs
        out["cgm_frac_below_70"] = (cgm < 70).sum(1) / n_obs
        out["min_since_above_180"] = _minutes_since(cgm > 180, dt_min)
        out["min_since_below_70"] = _minutes_since(cgm < 70, dt_min)
        k = f["meal_rise_minutes"] // dt_min
        # rise[t] = cgm[t] - min(cgm[t-k .. t]): how far glucose climbed in the last 30 min
        trailing_min = np.full_like(cgm_f, np.nan)
        trailing_min[:, k:] = sliding_window_view(cgm_f, k + 1, axis=1).min(-1)
        rise = cgm_f - trailing_min
        out["min_since_meal_rise"] = _minutes_since((rise >= f["meal_rise_mgdl"]) & obs, dt_min)
        out["max_rise_3h"] = np.nanmax(rise[:, -3 * sph:], 1)
        out["cgm_coverage"] = obs.mean(1)

        hours = (start_hour + np.asarray(anchor_idx) * dt_min / 60) % 24
        out["hour"] = hours
        out["tod_sin"], out["tod_cos"] = np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)

        out["steps_30m"] = np.nansum(steps[:, -sph // 2:], 1)
        out["steps_1h"] = np.nansum(steps[:, -sph:], 1)
        out["steps_3h"] = np.nansum(steps[:, -3 * sph:], 1)
        out["steps_max_1h"] = np.nanmax(steps[:, -sph:], 1)
        out["min_since_active"] = _minutes_since(steps >= f["active_steps_per_5min"], dt_min)
        out["wear_coverage"] = (~np.isnan(steps)).mean(1)
        out["hr_last"] = _ffill(hr)[:, -1]
        out["hr_mean_30m"] = np.nanmean(hr[:, -sph // 2:], 1)
        out["hr_minus_6h_mean"] = out["hr_mean_30m"] - np.nanmean(hr, 1)
        asleep = np.isin(sleep, (1, 2, 3))
        out["asleep_now"] = asleep[:, -1].astype(float)
        out["asleep_minutes"] = asleep.sum(1) * dt_min
        out["awake_in_bed_minutes"] = (sleep == 0).sum(1) * dt_min
        out["deep_frac"] = (sleep == 2).sum(1) / np.maximum(asleep.sum(1), 1)

    cols = [c for g in FEATURE_GROUPS.values() for c in g]
    return pd.DataFrame({c: np.asarray(out[c], dtype=np.float32) for c in cols})


def build_feature_table(series: np.ndarray, windows: pd.DataFrame, timestamps: pd.DatetimeIndex,
                        cfg: dict[str, Any], chunk: int = 20000) -> pd.DataFrame:
    """Handcrafted features for every row of `windows` (same order), computed in memory-bounded chunks."""
    history, _, _ = window_sizes(cfg)
    dt = cfg["sampling"]["interval_minutes"]
    start_hour = timestamps[0].hour + timestamps[0].minute / 60
    parts = []
    for a in range(0, len(windows), chunk):
        w = windows.iloc[a:a + chunk]
        raw = extract_windows(series, w.patient_idx.to_numpy(), w.anchor_idx.to_numpy(), history)
        parts.append(handcrafted_features(raw, w.anchor_idx.to_numpy(), start_hour, dt, cfg))
    return pd.concat(parts, ignore_index=True)
