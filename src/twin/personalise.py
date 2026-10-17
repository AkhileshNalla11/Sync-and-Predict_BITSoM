"""Personalised twin: estimate each patient's key simulator parameters from data a real system would have.

The oracle what-if replays the simulator with its HIDDEN parameters, schedule and noise, which no real
twin could do. This module builds the same mechanistic model from observable inputs only:

  * EHR record           -> population-default parameters via `derive_patient_params`, with the hidden
                            latent traits and random draws replaced by their population typical values
  * patient's own log    -> meals (type, carbs) and medication doses (events_truth.csv, filtered to the
                            event types a patient would log; see `LOGGED_EVENTS`)
  * wearable             -> steps, sleep (short nights raise next-day insulin resistance, as in the simulator)
  * first `fit_days` CGM -> four parameters fitted by penalised least squares (scipy least_squares, bounded trust-region Gauss-Newton):
        g_basal        untreated basal glucose
        meal_gain      mg/dL rise per absorbed gram (insulin resistance)
        tau_clear_min  glucose clearance time constant
        drug_sens      insulin / sulfonylurea sensitivity (only for patients on those drugs)
    Everything else stays at the EHR default. The penalty pulls weakly toward the EHR default (MAP-style).

Speed: the simulator is vectorised over patients, so candidate parameter sets are simulated side by side
(central-difference gradients cost one call). One patient fits in a few seconds; fits are cached in
`<processed_dir>/twin_fits.csv`.

Caveats: the log is exact here (synthetic), the model structure is the simulator's own structure, and the
noise model is known. Real carb counts, model mismatch and sensor quirks would make this much harder.
SYNTHETIC DATA, NOT CLINICALLY VALIDATED.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.stats import qmc

from src.config import resolve_path
from src.data.synthetic_gen import MEAL_CATALOGUE, _Schedule, derive_patient_params, simulate_glucose

FIT_VERSION = "1"
FIT_PARAMS = ("g_basal", "meal_gain", "tau_clear_min", "drug_sens")
# Prior SD (log scale) of each parameter around its EHR default, and search bounds (log scale).
PRIOR_SD = {"g_basal": 0.25, "meal_gain": 0.35, "tau_clear_min": 0.35, "drug_sens": 0.5}
BOUNDS = {"g_basal": (-0.7, 0.7), "meal_gain": (-1.2, 1.2), "tau_clear_min": (-1.2, 1.2), "drug_sens": (-1.2, 1.2)}
LOGGED_EVENTS = ("meal", "sulfonylurea_dose", "insulin_dose", "extra_insulin_dose")
SENSOR_LAG_MIN = 10            # CGM interstitial lag (same first-order lag as the simulator's sensor model)
SHORT_SLEEP_H = 5.5            # wearable-detected night shorter than this ...
SHORT_SLEEP_IR = 1.18          # ... raises next-day insulin resistance by the simulator's typical amount
N_COARSE = 64                  # candidates in the coarse batched search before the Gauss-Newton refinement
GRAD_STEP = 0.005              # central-difference step (log scale)


# --------------------------------------------------------------------------- EHR defaults
class _TypicalRng:
    """Stands in for a Generator: every draw returns the distribution's centre (mean / midpoint)."""

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any = None) -> np.ndarray:
        return np.full(size, float(loc))

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any = None) -> np.ndarray:
        return np.full(size, (low + high) / 2)


def ehr_default_params(ehr: pd.DataFrame, cal: dict[str, float], dt: int) -> pd.DataFrame:
    """Simulator parameters from the EHR alone: hidden traits at their population medians, no random draws."""
    ehr = ehr.reset_index() if "patient_id" not in ehr.columns else ehr.reset_index(drop=True)
    latent = pd.DataFrame({"patient_id": ehr.patient_id, "drug_sensitivity": 1.0, "ir_noise": 0.0})
    return derive_patient_params(ehr, latent, cal, dt, _TypicalRng())     # type: ignore[arg-type]


def on_glucose_lowering_drug(ehr_row: pd.Series) -> bool:
    return bool(ehr_row["on_insulin"]) or bool(ehr_row["on_sulfonylurea"])


# --------------------------------------------------------------------------- observable inputs
def load_patient_log(cfg: dict[str, Any]) -> pd.DataFrame:
    """Meals and medication doses as the patient would log them (cached as processed/patient_log.csv.gz)."""
    path = resolve_path(cfg["paths"]["processed_dir"]) / "patient_log.csv.gz"
    if not path.exists():
        ev = pd.read_csv(resolve_path(cfg["paths"]["raw_dir"]) / "events_truth.csv")
        ev = ev[ev.event_type.isin(LOGGED_EVENTS)][["patient_id", "timestamp", "event_type", "detail", "value"]]
        ev.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["timestamp"], keep_default_na=False, na_values=[""])


@dataclass
class PatientInputs:
    patient_id: str
    ehr_row: pd.Series
    series: np.ndarray            # (T_keep, 4) observed cgm, steps, hr, sleep_code
    timestamps: pd.DatetimeIndex
    log: pd.DataFrame             # this patient's logged meals and doses


@lru_cache(maxsize=4)
def _observed(processed_dir: str, raw_dir: str) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, int], pd.DataFrame]:
    npz = np.load(Path(processed_dir) / "series.npz", allow_pickle=False)
    ids = {str(p): i for i, p in enumerate(npz["patient_ids"])}
    ehr = pd.read_csv(Path(raw_dir) / "ehr.csv", keep_default_na=False).set_index("patient_id")
    return npz["series"], pd.DatetimeIndex(npz["timestamps"]), ids, ehr


@lru_cache(maxsize=4)
def _log_cached(cfg_json: str) -> pd.DataFrame:
    return load_patient_log(json.loads(cfg_json))


def patient_inputs(cfg: dict[str, Any], patient_id: str) -> PatientInputs:
    series, ts, ids, ehr = _observed(str(resolve_path(cfg["paths"]["processed_dir"])), str(resolve_path(cfg["paths"]["raw_dir"])))
    log = _log_cached(json.dumps({"paths": cfg["paths"]}, sort_keys=True))
    return PatientInputs(patient_id, ehr.loc[patient_id], series[ids[patient_id]], ts,
                         log[log.patient_id == patient_id].reset_index(drop=True))


def _sleep_episodes(sleep_code: np.ndarray, dt: int, max_gap_min: int = 30, min_len_h: float = 2.0) -> list[tuple[int, int]]:
    """(start, end) index pairs of wearable-detected sleep episodes; short gaps are bridged."""
    on = np.flatnonzero(~np.isnan(sleep_code))
    if not len(on):
        return []
    breaks = np.flatnonzero(np.diff(on) > max_gap_min // dt + 1)
    starts, ends = np.r_[on[0], on[breaks + 1]], np.r_[on[breaks], on[-1]] + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if (b - a) * dt / 60 >= min_len_h]


def schedule_from_log(inp: PatientInputs, dt: int, warm: int) -> _Schedule:
    """Driver arrays for the simulator from the log + wearable. Absolute steps include `warm` warm-up steps,
    which repeat the first observed day so drug and gut compartments start near steady state."""
    n_keep = len(inp.series)
    s = _Schedule(warm + n_keep, dt)
    t0 = inp.timestamps[0]
    for r in inp.log.itertuples():
        k = warm + int(round((r.timestamp - t0).total_seconds() / 60 / dt))
        if not 0 <= k < s.n:
            continue
        v = float(r.value)
        if r.event_type == "meal":
            s.carbs[MEAL_CATALOGUE[r.detail][2], k] += v
        elif r.event_type == "sulfonylurea_dose":
            s.doses[3, k] += v
        elif r.detail == "basal":
            s.doses[2, k] += v
        else:                                                    # premixed 30/70
            s.doses[0, k] += 0.3 * v
            s.doses[1, k] += 0.7 * v
        s.events.append({"step": k, "event_type": r.event_type, "detail": r.detail, "value": v})

    steps, sleep = inp.series[:, 1], inp.series[:, 3]
    s.worn[warm:] = ~np.isnan(steps)
    s.steps[warm:] = np.nan_to_num(steps)
    s.stage[warm:] = np.where(np.isnan(sleep), -1, sleep).astype(np.int8)
    for a, b in _sleep_episodes(sleep, dt):
        hours = (b - a) * dt / 60
        short = hours < SHORT_SLEEP_H and a > 0                  # an episode cut by the record start is not "short"
        s.events.append({"step": warm + a, "event_type": "sleep", "detail": "short" if short else "normal", "value": hours})
        s.ir_mult[warm + b: warm + b + 24 * 60 // dt] = SHORT_SLEEP_IR if short else 1.0
    s.events.sort(key=lambda e: e["step"])

    day = min(warm, n_keep)                                      # warm-up = copy of the first observed day
    for arr in (s.carbs, s.doses):
        arr[:, warm - day: warm] = arr[:, warm: warm + day]
    for arr in (s.steps, s.ir_mult, s.stage, s.worn):
        arr[warm - day: warm] = arr[warm: warm + day]
    return s


# --------------------------------------------------------------------------- simulation helpers
def simulate_schedule(params: pd.DataFrame, s: _Schedule, dt: int, cal: dict[str, float], n_steps: int | None = None,
                      noise: np.ndarray | None = None, obs: np.ndarray | None = None, obs_gain: float = 0.0,
                      corrections: np.ndarray | None = None) -> np.ndarray:
    """Simulate one schedule for P parameter sets side by side (deterministic: no process noise). (P, T)."""
    n_p, n_t = len(params), n_steps or s.n
    tod = (np.arange(n_t) * dt / 60) % 24
    return simulate_glucose(params.reset_index(drop=True),
                            np.broadcast_to(s.carbs[:, None, :n_t], (3, n_p, n_t)),
                            np.broadcast_to(s.doses[:, None, :n_t], (4, n_p, n_t)),
                            np.broadcast_to(s.steps[None, :n_t], (n_p, n_t)),
                            np.broadcast_to(s.ir_mult[None, :n_t], (n_p, n_t)),
                            tod, dt, cal, None, noise=np.zeros((n_t, n_p)) if noise is None else noise,
                            obs=obs, obs_gain=obs_gain, corrections=corrections)


def sensor_view(g: np.ndarray, dt: int) -> np.ndarray:
    """Noise-free CGM: the simulator's first-order interstitial lag and 40-400 mg/dL limits, without rounding."""
    a = dt / SENSOR_LAG_MIN
    lagged, _ = lfilter([a], [1.0, a - 1.0], g, axis=-1, zi=(1 - a) * g[..., :1])
    return np.clip(lagged, 40, 400)


def observer_target(cgm: np.ndarray, warm: int, n_total: int, until: int, dt: int) -> np.ndarray:
    """Glucose observations for the state observer, using CGM up to kept index `until` only.

    CGM lags blood glucose by ~SENSOR_LAG_MIN, so glucose at step t is estimated by the reading at t + lag.
    """
    lag = int(round(SENSOR_LAG_MIN / dt))
    out = np.full(n_total, np.nan)
    i = np.arange(0, max(0, until - lag + 1))
    out[warm + i] = cgm[i + lag]
    return out


# --------------------------------------------------------------------------- fitting
@dataclass
class FitResult:
    params: pd.DataFrame          # one row: full simulator parameters (fitted subset + EHR defaults)
    default: pd.DataFrame         # one row: EHR-default parameters
    info: dict[str, Any]          # loss, MAE on the fit window, observations used, evaluations, seconds


def settings(cfg: dict[str, Any], n_days: int) -> dict[str, Any]:
    p = cfg["twin"]["personalise"]
    return {"fit_days": int(max(1, min(p["fit_days"], n_days // 2))), "prior_weight": float(p["prior_weight"]),
            "observer_gain": float(p["observer_gain"])}


def fit_parameters(s: _Schedule, cgm: np.ndarray, default: pd.DataFrame, cal: dict[str, float], dt: int, warm: int,
                   fit_steps: int, fit_drug: bool, prior_weight: float, seed: int = 0) -> FitResult:
    """Penalised least squares of noise-free simulated CGM vs observed CGM over the first `fit_steps`.

    Objective: mean squared CGM error + prior_weight * sum((log(theta / theta_ehr) / PRIOR_SD)^2).
    """
    t_start = time.time()
    names = [n for n in FIT_PARAMS if fit_drug or n != "drug_sens"]
    sd = np.array([PRIOR_SD[n] for n in names])
    lo, hi = np.array([BOUNDS[n][0] for n in names]), np.array([BOUNDS[n][1] for n in names])
    base = default.iloc[0]
    obs = cgm[:fit_steps]
    mask = ~np.isnan(obs)
    if mask.sum() < 24:
        raise ValueError("too few CGM readings in the fit window")
    n_steps = warm + fit_steps
    n_calls = 0

    def candidates(z: np.ndarray) -> pd.DataFrame:
        rows = default.loc[default.index.repeat(len(z))].reset_index(drop=True)
        for j, n in enumerate(names):
            rows[n] = float(base[n]) * np.exp(z[:, j])
        return rows

    w_obs, w_prior = 1 / np.sqrt(mask.sum()), np.sqrt(prior_weight) / sd

    def residuals(z: np.ndarray) -> np.ndarray:
        """(P, n_obs + k): CGM errors scaled so that sum of squares = MSE, then the prior terms."""
        nonlocal n_calls
        n_calls += 1
        sim = sensor_view(simulate_schedule(candidates(z), s, dt, cal, n_steps), dt)[:, warm:]
        return np.hstack([(sim[:, mask] - obs[mask]) * w_obs, z * w_prior])

    # 1) coarse batched search over the box (one simulator call)
    z0 = qmc.scale(qmc.Sobol(len(names), scramble=True, seed=seed).random(N_COARSE), lo, hi)
    z0[0] = 0.0                                                   # always include the EHR default
    start = z0[int(np.argmin((residuals(z0) ** 2).sum(1)))]

    # 2) bounded Gauss-Newton (trust-region reflective). Residuals and the central-difference Jacobian
    #    come from one batched call; the Jacobian is cached for the point least_squares asks about next.
    eye = np.eye(len(names)) * GRAD_STEP
    cache: dict[bytes, np.ndarray] = {}

    def fun(z: np.ndarray) -> np.ndarray:
        r = residuals(np.vstack([z, z + eye, z - eye]))
        k = len(names)
        cache.clear()
        cache[z.tobytes()] = ((r[1:k + 1] - r[k + 1:]) / (2 * GRAD_STEP)).T
        return r[0]

    def jac(z: np.ndarray) -> np.ndarray:
        if z.tobytes() not in cache:
            fun(z)
        return cache[z.tobytes()]

    res = least_squares(fun, np.clip(start, lo + 1e-6, hi - 1e-6), jac=jac, bounds=(lo, hi), method="trf",
                        x_scale=sd, max_nfev=12, ftol=1e-4, xtol=1e-4)
    z = np.clip(res.x, lo, hi)
    fitted = candidates(z[None])
    sim = sensor_view(simulate_schedule(fitted, s, dt, cal, n_steps), dt)[0, warm:]
    info = {"loss": float(2 * res.cost), "converged": bool(res.status > 0), "fit_mae": float(np.mean(np.abs(sim[mask] - obs[mask]))),
            "n_obs": int(mask.sum()), "n_calls": n_calls + 1, "seconds": round(time.time() - t_start, 2),
            "fitted_params": ",".join(names), **{f"z_{n}": float(v) for n, v in zip(names, z)}}
    return FitResult(params=fitted, default=default, info=info)


def fit_patient(cfg: dict[str, Any], patient_id: str, inp: PatientInputs | None = None) -> FitResult:
    inp = inp or patient_inputs(cfg, patient_id)
    dt = cfg["sampling"]["interval_minutes"]
    cal = cfg["simulation"]["calibration"]
    warm = cfg["simulation"]["warmup_days"] * 24 * 60 // dt
    st = settings(cfg, len(inp.series) * dt // (24 * 60))
    default = ehr_default_params(pd.DataFrame([{**inp.ehr_row.to_dict(), "patient_id": patient_id}]), cal, dt)
    s = schedule_from_log(inp, dt, warm)
    fit = fit_parameters(s, inp.series[:, 0], default, cal, dt, warm, st["fit_days"] * 24 * 60 // dt,
                         on_glucose_lowering_drug(inp.ehr_row), st["prior_weight"], seed=cfg["seed"])
    fit.info.update(fit_days=st["fit_days"], prior_weight=st["prior_weight"])
    return fit


# --------------------------------------------------------------------------- cache
def _cache_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["paths"]["processed_dir"]) / "twin_fits.csv"


def _cache_key(cfg: dict[str, Any], n_days: int) -> dict[str, Any]:
    st = settings(cfg, n_days)
    return {"fit_version": FIT_VERSION, "fit_days": st["fit_days"], "prior_weight": st["prior_weight"]}


def _matches(row: pd.Series, key: dict[str, Any]) -> bool:
    """Is a cached fit current? (CSV round trips turn ints into floats, so compare numbers as numbers.)"""
    try:
        return str(row.get("fit_version")) == key["fit_version"] and all(
            float(row.get(k)) == float(v) for k, v in key.items() if k != "fit_version")
    except (TypeError, ValueError):
        return False


def load_fit_cache(cfg: dict[str, Any]) -> pd.DataFrame:
    path = _cache_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=["patient_id"])
    return pd.read_csv(path, dtype={"fit_version": str})


def get_fit(cfg: dict[str, Any], patient_id: str, refit: bool = False) -> FitResult:
    """Fitted parameters for one patient: from the cache if current, else fit now (seconds) and cache."""
    inp = patient_inputs(cfg, patient_id)
    dt = cfg["sampling"]["interval_minutes"]
    key = _cache_key(cfg, len(inp.series) * dt // (24 * 60))
    cache = load_fit_cache(cfg)
    row = cache[cache.patient_id == patient_id]
    if not refit and len(row) and _matches(row.iloc[0], key):
        r = row.iloc[0]
        default = ehr_default_params(pd.DataFrame([{**inp.ehr_row.to_dict(), "patient_id": patient_id}]),
                                     cfg["simulation"]["calibration"], dt)
        fitted = default.copy()
        for n in FIT_PARAMS:
            fitted[n] = float(r[n])
        info = {k: r[k] for k in r.index if k not in FIT_PARAMS and k != "patient_id"}
        return FitResult(params=fitted, default=default, info=info)
    fit = fit_patient(cfg, patient_id, inp)
    save_fits(cfg, {patient_id: fit})
    return fit


def save_fits(cfg: dict[str, Any], fits: dict[str, FitResult]) -> None:
    rows = [{"patient_id": pid, **{n: float(f.params.iloc[0][n]) for n in FIT_PARAMS},
             **{f"ehr_default_{n}": float(f.default.iloc[0][n]) for n in FIT_PARAMS}, **f.info,
             "fit_version": FIT_VERSION} for pid, f in fits.items()]
    cache = load_fit_cache(cfg)
    cache = pd.concat([cache[~cache.patient_id.isin(fits)], pd.DataFrame(rows)], ignore_index=True)
    try:
        cache.sort_values("patient_id").to_csv(_cache_path(cfg), index=False)
    except OSError:                                              # read-only deployment: keep the fit in memory only
        pass
