"""Mechanistic generator for synthetic Type 2 Diabetes wearable + CGM data.

SYNTHETIC DATA, NOT CLINICALLY VALIDATED. The model is a deliberately simple,
Bergman-minimal-model-inspired system at 5-minute resolution. It is designed to
produce *plausible-looking* dynamics with known cause and effect, so that the
downstream ML models have something real to learn. It is not a physiological
simulator and must not be used to reason about real patients.

Per patient and per 5-minute step:
    dG = -p1 (G - Gb_eff)                 return toward basal glucose
         - X_drug * G                     insulin / sulfonylurea effect
         - X_ex * max(G - 75, 0)          activity uptake (cannot by itself cause lows)
         + gain * ir_day * Ra             carbohydrate appearance from meals
         + cr * max(75 - G, 0)            counter-regulation (weaker with long duration)
         + noise
  * Meals pass through a 2-compartment gut chain; absorption speed depends on
    meal type (fast: rice, idli, sweets; medium: roti, poha; slow: paratha).
  * Drugs pass through 2-compartment PK chains: rapid insulin, NPH (premix
    30/70 peaks ~5 h after dose -> nocturnal lows after the evening dose),
    basal insulin, sulfonylurea. Low eGFR slows clearance.
  * Recent activity amplifies drug effect (post-exercise insulin sensitivity).
  * A poor night's sleep raises next-day insulin resistance.
  * Dawn phenomenon raises basal glucose around 04:00-08:00.
The observed CGM adds a 10-min interstitial lag, autocorrelated sensor error,
per-sensor calibration bias, 40-400 mg/dL reporting limits, and dropouts.

Hidden simulator state (true glucose, meals, doses, latent traits) is written to
separate *_truth / *_params files for auditing and the what-if engine; it is
never used as a model input.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.ehr_features import generate_cohort
from src.data.wearable_features import label_windows

# Indian-style meal catalogue: name -> (carb grams low, high, absorption class).
# Portions are rough typical values, not nutritional ground truth.
MEAL_CATALOGUE: dict[str, tuple[float, float, int]] = {
    "rice_meal": (75, 110, 0),       # rice + dal + sabzi
    "idli_dosa": (50, 70, 0),
    "sweets": (30, 50, 0),           # gulab jamun, jalebi, ladoo
    "tea_biscuits": (15, 25, 0),
    "roti_meal": (50, 75, 1),        # 2-3 roti + dal + sabzi
    "poha_upma": (40, 60, 1),
    "fruit": (15, 25, 1),
    "paratha": (50, 70, 2),          # fat slows gastric emptying
}
ABSORPTION_TAU_MIN = np.array([30.0, 45.0, 65.0])      # fast, medium, slow

DRUGS = ("rapid", "nph", "basal", "su")
DRUG_TAU_MIN = np.array([75.0, 300.0, 600.0, 300.0])   # glimepiride acts ~24 h
# Glucose-lowering effect per unit (insulin) / per mg (sulfonylurea) in the effect compartment.
DRUG_POTENCY = np.array([0.0060, 0.0050, 0.0035, 0.0300])

SLEEP_STAGES = {-1: None, 0: "awake", 1: "light", 2: "deep", 3: "rem"}


# --------------------------------------------------------------------------- params
def nominal_mean_drug_effect(ehr: pd.DataFrame, cal: dict[str, float]) -> np.ndarray:
    """Average drug effect a prescriber would expect (nominal sensitivity, normal kidneys).

    For a 2-compartment chain fed a constant daily dose D, the mean amount in the
    effect compartment is D * tau / 1440 (tau in minutes).
    """
    tdd = ehr.insulin_total_daily_units.to_numpy(dtype=float)
    premix = (ehr.insulin_regimen == "premix_bid").to_numpy()
    basal = (ehr.insulin_regimen == "basal").to_numpy()
    daily = np.stack([np.where(premix, 0.3 * tdd, 0), np.where(premix, 0.7 * tdd, 0),
                      np.where(basal, tdd, 0), ehr.sulfonylurea_dose_mg.to_numpy(dtype=float)])
    return cal["drug_effect_scale"] * (DRUG_POTENCY[:, None] * daily * DRUG_TAU_MIN[:, None] / 1440).sum(0)


def derive_patient_params(ehr: pd.DataFrame, latent: pd.DataFrame, cal: dict[str, float], dt: int,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Map EHR traits (+ hidden latents) to simulator parameters. This is where fusion signal is created.

    HbA1c is an *on-treatment* lab, so it sets the treated basal glucose `g_target`. Drug users get a
    higher untreated basal `g_basal`, chosen so the drug effect the prescriber *expects* brings them back
    to `g_target` on average (titration). Titration is partial: the prescriber has seen sqrt(sensitivity)
    of the patient's true response, but does not adjust for kidney function or age. Lows then come from
    timing mismatches (dose peaks, skipped or delayed meals, exercise, extra doses) and from the residual
    hidden over-exposure. Drug dependence (untreated / treated basal - 1) is capped so no patient's
    glucose is a fragile balance of two huge opposing terms.
    """
    n = len(ehr)
    a1c, bmi, egfr = ehr.hba1c.to_numpy(), ehr.bmi.to_numpy(), ehr.egfr.to_numpy()
    dur, age = ehr.diabetes_duration_years.to_numpy(), ehr.age.to_numpy()

    ir = np.exp(0.22 * (a1c - 7.5) + 0.03 * (bmi - 26.5) + latent.ir_noise.to_numpy())
    tau_clear = 80 + 30 * ir
    g_target = 104 + 18 * (a1c - 6.5) + rng.normal(0, 6, n)
    g_target = np.clip(np.where(ehr.on_metformin, g_target * 0.95, g_target), 90, 230)
    renal = 1 + 0.6 * np.clip((60 - egfr) / 45, 0, 1)            # eGFR 15 -> 1.6x slower drug clearance
    latent_sens = latent.drug_sensitivity.to_numpy()
    drug_sens = latent_sens * (1 + 0.5 * (renal - 1)) * (1 + 0.01 * np.clip(age - 60, 0, None))

    dependence = nominal_mean_drug_effect(ehr, cal) / (dt / tau_clear) * latent_sens ** cal["titration_exponent"]
    potency_mult = np.minimum(1.0, cal["max_drug_dependence"] / np.maximum(dependence, 1e-9))
    g_basal = g_target * (1 + dependence * potency_mult)
    drug_sens = drug_sens * potency_mult

    return pd.DataFrame({
        "patient_id": ehr.patient_id,
        "insulin_resistance": ir,
        "g_target": g_target,
        "g_basal": g_basal,
        "tau_clear_min": tau_clear,
        "meal_gain": ir,
        "drug_sens": drug_sens,
        "renal_factor": renal,
        "cr_strength": 0.25 * np.exp(-0.04 * dur) * rng.uniform(0.8, 1.2, n),
        "dawn_amp": np.clip(4 + 7 * (a1c - 6.5) + rng.normal(0, 4, n), 0, 40),
        "hr_rest": np.clip(72 + rng.normal(0, 6, n) + 3 * ehr.hypertension.to_numpy(), 55, 95),
        "steps_base": rng.uniform(10, 35, n) * np.clip(1 - 0.015 * (bmi - 26.5), 0.6, 1.3),
        "fitness": np.clip(rng.normal(1.0, 0.1, n), 0.75, 1.25),
        "morning_walk_prob": np.clip(rng.normal(0.45, 0.25, n) - 0.02 * (bmi - 26.5), 0, 0.95),
        "evening_walk_prob": np.clip(rng.normal(0.30, 0.20, n), 0, 0.9),
        "poor_sleep_prob": np.clip(rng.normal(0.15, 0.08, n), 0.02, 0.45),
        "late_dinner_prob": np.clip(rng.normal(0.15, 0.10, n), 0, 0.5),
        "rice_pref": rng.uniform(0, 1, n),
        "portion": np.clip(rng.normal(1.0, 0.12, n), 0.7, 1.4),
        "wake_mean_h": rng.normal(6.3, 0.5, n),
        "sleep_onset_mean_h": rng.normal(22.8, 0.6, n),
    })


# --------------------------------------------------------------------------- schedule
class _Schedule:
    """Per-patient driver arrays for the whole simulated period (incl. warm-up)."""

    def __init__(self, n_steps: int, dt: int):
        self.dt = dt
        self.n = n_steps
        self.carbs = np.zeros((3, n_steps))
        self.doses = np.zeros((4, n_steps))
        self.steps = np.zeros(n_steps)
        self.ir_mult = np.ones(n_steps)
        self.stage = np.full(n_steps, -1, dtype=np.int8)
        self.worn = np.ones(n_steps, dtype=bool)
        self.events: list[dict[str, Any]] = []

    def idx(self, abs_hour: float) -> int:
        return int(np.clip(round(abs_hour * 60 / self.dt), 0, self.n - 1))

    def span(self, start_h: float, dur_min: float) -> slice:
        a = self.idx(start_h)
        return slice(a, min(self.n, a + max(1, int(round(dur_min / self.dt)))))

    def log(self, abs_hour: float, etype: str, detail: str, value: float) -> None:
        self.events.append({"step": self.idx(abs_hour), "event_type": etype, "detail": detail, "value": round(float(value), 2)})


def _fill_sleep(s: _Schedule, onset_h: float, wake_h: float, poor: bool, rng: np.random.Generator) -> None:
    """Simple 90-minute cycle model: deep sleep early in the night, REM later, wake bouts."""
    sl = s.span(onset_h, (wake_h - onset_h) * 60)
    n = sl.stop - sl.start
    j = np.arange(n)
    cycle, pos = j // 18, (j % 18) / 18
    deep_p = np.clip((0.55 if poor else 0.85) - 0.2 * cycle, 0, 1)
    u = rng.random(n)
    stage = np.ones(n, dtype=np.int8)
    stage[(pos > 0.15) & (pos < 0.6) & (u < deep_p)] = 2
    stage[(pos > 0.62) & (u < 0.45 + 0.1 * cycle)] = 3
    wake_start = rng.random(n) < (0.10 if poor else 0.025)
    for k in np.flatnonzero(wake_start):
        stage[k:k + rng.integers(1, 4)] = 0
    s.stage[sl] = stage


def _pick_meal(rng: np.random.Generator, options: dict[str, float]) -> str:
    names = list(options)
    p = np.array(list(options.values()))
    return str(rng.choice(names, p=p / p.sum()))


def generate_schedule(p: pd.Series, ehr: pd.Series, n_days: int, dt: int,
                      rng: np.random.Generator) -> _Schedule:
    """Daily life of one virtual patient: sleep, meals, doses, activity, device wear."""
    s = _Schedule(n_days * 24 * 60 // dt, dt)
    on_su, on_ins = bool(ehr.on_sulfonylurea), bool(ehr.on_insulin)
    premix = ehr.insulin_regimen == "premix_bid"
    tdd = float(ehr.insulin_total_daily_units)

    late_dinner = rng.random(n_days) < p.late_dinner_prob
    poor_night = rng.random(n_days + 1) < p.poor_sleep_prob     # index d+1 = night after day d
    extra_activity = rng.random(n_days) < 0.07
    extra_dose = rng.random(n_days) < 0.02

    # ---- sleep: night k runs from day k-1 evening to day k morning (k = 0 is the night before day 0)
    wake_abs = np.zeros(n_days + 1)
    onset_abs = np.zeros(n_days + 1)
    for k in range(n_days + 1):
        d = k - 1
        poor = bool(poor_night[k])
        onset = 24 * d + p.sleep_onset_mean_h + rng.normal(0, 0.45)
        onset += (0.75 if d >= 0 and late_dinner[d] else 0) + (rng.uniform(0.5, 1.5) if poor else 0)
        dur = rng.uniform(4.0, 5.5) if poor else float(np.clip(rng.normal(7.2, 0.5), 5.5, 9.0))
        onset_abs[k], wake_abs[k] = onset, onset + dur
        if wake_abs[k] > 0:
            _fill_sleep(s, max(onset, 0), wake_abs[k], poor, rng)
            s.log(max(onset, 0), "sleep", "poor" if poor else "normal", dur)
        # next-day insulin resistance depends on the night's sleep
        nxt = slice(s.idx(wake_abs[k]), s.idx(24 * (k + 1) + p.wake_mean_h))
        s.ir_mult[nxt] = 1 + (rng.uniform(0.12, 0.25) if poor else rng.normal(0, 0.03))
        if rng.random() < 0.06:                                  # watch not worn overnight
            s.worn[s.span(max(onset, 0), dur * 60)] = False

    for d in range(n_days):
        base = 24 * d
        wake, bed = wake_abs[d], onset_abs[d + 1]
        day_scale = rng.normal(1, 0.08)

        def eat(abs_h: float, name: str) -> None:
            lo, hi, cls = MEAL_CATALOGUE[name]
            g = rng.uniform(lo, hi) * p.portion * day_scale
            s.carbs[cls, s.idx(abs_h)] += g
            s.log(abs_h, "meal", name, g)

        def main_meal(planned: float, name: str, p_skip: float, p_delay: float) -> tuple[float, str]:
            """Returns (actual time or NaN if skipped, status)."""
            u = rng.random()
            if u < p_skip:
                s.log(planned, "meal_skipped", name, 0)
                return np.nan, "skipped"
            if u < p_skip + p_delay:
                delay = rng.uniform(1.0, 2.5)
                s.log(planned, "meal_delayed", name, delay)
                eat(planned + delay, name)
                return planned + delay, "delayed"
            eat(planned, name)
            return planned, "on_time"

        r = p.rice_pref
        bf_plan = max(wake + 0.9, base + 7.0) + rng.normal(0, 0.25)
        bf_type = _pick_meal(rng, {"idli_dosa": 0.2 + 0.5 * r, "poha_upma": 0.3,
                                   "paratha": 0.2 + 0.4 * (1 - r), "roti_meal": 0.1})
        bf_t, bf_status = main_meal(bf_plan, bf_type, 0.07, 0.06)
        lunch_plan = base + 13.5 + rng.normal(0, 0.4)
        main_meal(lunch_plan, "rice_meal" if rng.random() < r else "roti_meal", 0.04, 0.10)
        if rng.random() < 0.75:
            eat(base + 17.25 + rng.normal(0, 0.4), _pick_meal(rng, {"tea_biscuits": 0.7, "sweets": 0.15, "fruit": 0.15}))
        if late_dinner[d]:
            dn_plan = base + 22.3 + rng.uniform(0, 1.0)
            s.log(dn_plan, "late_dinner", "", dn_plan - base)
        else:
            dn_plan = base + 20.75 + rng.normal(0, 0.4)
        dn_t, dn_status = main_meal(dn_plan, "rice_meal" if rng.random() < 0.8 * r else "roti_meal", 0.03, 0.05)
        if dn_status != "skipped" and rng.random() < 0.12:
            eat(dn_t + 0.33, "sweets")

        # ---- medication. Doses follow the *planned* meal time unless the patient waits for the meal;
        # a dose without its meal is the main driver of daytime lows.
        def dose_time(planned: float, actual: float, status: str, p_take_if_skipped: float) -> float:
            if status == "skipped":
                return planned if rng.random() < p_take_if_skipped else np.nan
            if status == "delayed":
                return planned if rng.random() < 0.5 else actual
            return actual

        if on_su:
            bd = ehr.sulfonylurea_frequency == "BD"
            slots = [(0.5, bf_plan, bf_t, bf_status), (0.5, dn_plan, dn_t, dn_status)] if bd else [(1.0, bf_plan, bf_t, bf_status)]
            for frac, planned, actual, status in slots:
                t = dose_time(planned - 0.25, actual, status, 0.8)
                if np.isnan(t) or rng.random() < 0.04:            # missed dose
                    continue
                double = extra_dose[d] and planned == bf_plan
                mg = float(ehr.sulfonylurea_dose_mg) * frac * (2 if double else 1)
                s.doses[3, s.idx(t)] += mg
                s.log(t, "sulfonylurea_dose", "double" if double else "", mg)
        if on_ins and premix:
            for frac, planned, actual, status in ((0.5, bf_plan, bf_t, bf_status), (0.5, dn_plan, dn_t, dn_status)):
                t = dose_time(planned, actual, status, 0.4)
                if np.isnan(t) or rng.random() < 0.03:
                    continue
                units = tdd * frac * rng.normal(1, 0.05)
                s.doses[0, s.idx(t)] += 0.3 * units
                s.doses[1, s.idx(t)] += 0.7 * units
                s.log(t, "insulin_dose", "premix_30_70", units)
            if extra_dose[d] and dn_status != "skipped":     # repeated evening injection
                units = tdd * 0.5 * rng.uniform(0.5, 1.0)
                t = dn_t + 1.0
                s.doses[0, s.idx(t)] += 0.3 * units
                s.doses[1, s.idx(t)] += 0.7 * units
                s.log(t, "extra_insulin_dose", "premix_30_70", units)
        elif on_ins:
            t = base + 22.0 + rng.normal(0, 0.3)
            if rng.random() > 0.03:
                units = tdd * rng.normal(1, 0.03) * (1.5 if extra_dose[d] else 1)
                s.doses[2, s.idx(t)] += units
                s.log(t, "extra_insulin_dose" if extra_dose[d] else "insulin_dose", "basal", units)

        # ---- activity
        awake = s.span(wake, (bed - wake) * 60)
        s.steps[awake] += rng.poisson(p.steps_base, awake.stop - awake.start)

        def walk(start: float, dur_min: float, rate: float, label: str) -> None:
            sl = s.span(start, dur_min)
            s.steps[sl] += rng.poisson(max(rate, 50) * p.fitness, sl.stop - sl.start)
            s.log(start, "activity", label, dur_min)

        if rng.random() < p.morning_walk_prob:
            walk(wake + rng.uniform(0.2, 0.6), rng.uniform(20, 50), rng.normal(480, 50), "morning_walk")
        if dn_status != "skipped" and rng.random() < p.evening_walk_prob:
            walk(dn_t + rng.uniform(0.2, 0.5), rng.uniform(15, 30), rng.normal(350, 40), "evening_walk")
        for _ in range(rng.poisson(1.5)):
            walk(rng.uniform(wake + 1, bed - 1), rng.uniform(5, 20), rng.uniform(200, 300), "errand")
        if extra_activity[d]:
            walk(base + rng.uniform(9.5, 16.5), rng.uniform(60, 100), rng.uniform(450, 550), "extra_activity")

        if rng.random() < 0.8:                                   # watch off for charging / bathing
            s.worn[s.span(rng.uniform(wake + 0.5, bed - 1.5), rng.uniform(45, 90))] = False
    return s


# --------------------------------------------------------------------------- dynamics
def _chain_step(q: np.ndarray, inp: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Move input and compartment-1 outflow along 2-compartment chains; returns compartment 2 (a view)."""
    q[:, 0] += inp
    f1 = k * q[:, 0]
    q[:, 0] -= f1
    q[:, 1] += f1
    return q[:, 1]


def simulate_glucose(params: pd.DataFrame, carbs: np.ndarray, doses: np.ndarray, steps: np.ndarray,
                     ir_mult: np.ndarray, tod_hours: np.ndarray, dt: int, cal: dict[str, float],
                     rng: np.random.Generator) -> np.ndarray:
    """Integrate true glucose for P patients. carbs (3,P,T), doses (4,P,T), steps/ir_mult (P,T)."""
    n_p, n_t = steps.shape
    gb = params.g_basal.to_numpy()
    p1 = dt / params.tau_clear_min.to_numpy()
    gain = params.meal_gain.to_numpy() * cal["meal_gain"]
    sens = params.drug_sens.to_numpy() * cal["drug_effect_scale"]
    cr = params.cr_strength.to_numpy()
    dawn = params.dawn_amp.to_numpy()
    k_gut = (dt / ABSORPTION_TAU_MIN)[:, None]
    k_drug = dt / (DRUG_TAU_MIN[:, None] * params.renal_factor.to_numpy()[None, :])
    potency = DRUG_POTENCY[:, None]
    dawn_curve = np.exp(-0.5 * ((tod_hours - 6.5) / 1.3) ** 2)
    # Early night (peak ~02:30): insulin needs are lowest and sleep blunts counter-regulation,
    # which is why drug-treated patients are prone to nocturnal lows.
    night_curve = np.exp(-0.5 * ((((tod_hours + 12) % 24) - 14.5) / 2.0) ** 2)
    post_decay = np.exp(-dt / 180)

    g = gb.copy()
    gut = np.zeros((3, 2, n_p))
    pk = np.zeros((4, 2, n_p))
    s_post = np.zeros(n_p)
    noise = rng.normal(0, cal["process_noise_sd"], (n_t, n_p))
    out = np.empty((n_p, n_t))
    for t in range(n_t):
        q2 = _chain_step(gut, carbs[:, :, t], k_gut)
        ra = (k_gut * q2).sum(0)                     # grams absorbed this step
        gut[:, 1] -= k_gut * q2
        _chain_step(pk, doses[:, :, t], k_drug)
        drug_amt = pk[:, 1].copy()
        pk[:, 1] -= k_drug * pk[:, 1]

        act = np.minimum(steps[:, t] / 500, 1.5)
        s_post = s_post * post_decay + 0.12 * act
        irm = ir_mult[:, t]
        x_drug = ((potency * drug_amt).sum(0) * sens / np.sqrt(irm)
                  * (1 + cal["exercise_drug_boost"] * np.minimum(s_post, 1.5)) * (1 + 0.3 * night_curve[t]))
        x_ex = cal["exercise_scale"] * act
        gb_eff = gb * (1 + 0.3 * (irm - 1)) + dawn * dawn_curve[t]
        cr_t = cr * (1 - 0.5 * night_curve[t])
        dg = (-p1 * (g - gb_eff) - x_drug * g - x_ex * np.maximum(g - 75, 0)
              + gain * irm * ra + cr_t * np.maximum(75 - g, 0) + noise[t])
        g = np.clip(g + dg, 30, 500)
        out[:, t] = g
    return out


def apply_cgm_sensor(g_true: np.ndarray, dt: int, rng: np.random.Generator) -> np.ndarray:
    """Observed CGM: interstitial lag, AR(1) error, calibration bias, range limits, dropouts."""
    n_p, n_t = g_true.shape
    lagged = np.empty_like(g_true)
    s = g_true[:, 0].copy()
    err = np.zeros(n_p)
    phi, sd = 0.85, 5.0
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), (n_t, n_p))
    errs = np.empty_like(g_true)
    for t in range(n_t):
        s += dt / 10 * (g_true[:, t] - s)
        err = phi * err + e[t]
        lagged[:, t], errs[:, t] = s, err
    gain = rng.normal(1.0, 0.04, (n_p, 1))
    cgm = np.clip(np.round(lagged * gain + errs + rng.normal(0, 1.5, g_true.shape)), 40, 400)

    per_day = 24 * 60 // dt
    mask = rng.random(g_true.shape) < 0.01                       # isolated missed readings
    for i in range(n_p):
        for d in range(n_t // per_day):
            if rng.random() < 0.35:                              # signal-loss gap, 15-120 min
                a = d * per_day + rng.integers(0, per_day)
                mask[i, a:a + rng.integers(3, 25)] = True
        if rng.random() < 0.15:                                  # one long sensor failure, 6-12 h
            a = rng.integers(0, n_t)
            mask[i, a:a + rng.integers(72, 145)] = True
    cgm[mask] = np.nan
    return cgm


def simulate_heart_rate(params: pd.DataFrame, steps: np.ndarray, stage: np.ndarray, ir_mult: np.ndarray,
                        g_true: np.ndarray, dt: int, rng: np.random.Generator) -> np.ndarray:
    """HR from rest level, activity, sleep stage, poor-sleep strain and adrenergic response to lows."""
    n_p, n_t = steps.shape
    smooth = np.empty_like(steps)
    acc = np.zeros(n_p)
    a = dt / 10
    for t in range(n_t):
        acc += a * (steps[:, t] - acc)
        smooth[:, t] = acc
    sleep_offset = np.select([stage == 0, stage == 1, stage == 2, stage == 3], [-2.0, -7.0, -10.0, -4.0], 5.0)
    hr = (params.hr_rest.to_numpy()[:, None] + sleep_offset + 0.06 * smooth
          + 12 * (ir_mult - 1) + 0.35 * np.maximum(72 - g_true, 0))
    noise = np.zeros(n_p)
    out = np.empty_like(hr)
    eps = rng.normal(0, 2.0, (n_t, n_p))
    for t in range(n_t):
        noise = 0.7 * noise + eps[t]
        out[:, t] = hr[:, t] + noise
    return np.round(np.clip(out, 40, 190))


# --------------------------------------------------------------------------- orchestration
def generate_dataset(cfg: dict[str, Any], n_patients: int | None = None, n_days: int | None = None) -> dict[str, pd.DataFrame]:
    """Generate all tables. Same config + seed -> identical output."""
    sim = cfg["simulation"]
    seed = cfg["seed"]
    dt = cfg["sampling"]["interval_minutes"]
    n_patients = n_patients or sim["n_patients"]
    n_days = n_days or sim["n_days"]
    warm = sim["warmup_days"]
    total_days = n_days + warm

    ehr, latent = generate_cohort(n_patients, np.random.default_rng([seed, 0]))
    params = derive_patient_params(ehr, latent, sim["calibration"], dt, np.random.default_rng([seed, 1]))

    schedules = [generate_schedule(params.iloc[i], ehr.iloc[i], total_days, dt, np.random.default_rng([seed, 2, i]))
                 for i in range(n_patients)]
    carbs = np.stack([s.carbs for s in schedules], axis=1)
    doses = np.stack([s.doses for s in schedules], axis=1)
    steps = np.stack([s.steps for s in schedules])
    ir_mult = np.stack([s.ir_mult for s in schedules])
    stage = np.stack([s.stage for s in schedules])
    worn = np.stack([s.worn for s in schedules])
    n_t = steps.shape[1]
    tod = (np.arange(n_t) * dt / 60) % 24

    g_true = simulate_glucose(params, carbs, doses, steps, ir_mult, tod, dt, sim["calibration"],
                              np.random.default_rng([seed, 3]))
    cgm = apply_cgm_sensor(g_true, dt, np.random.default_rng([seed, 4]))
    hr = simulate_heart_rate(params, steps, stage, ir_mult, g_true, dt, np.random.default_rng([seed, 5]))

    keep = slice(warm * 24 * 60 // dt, n_t)
    n_keep = n_t - keep.start
    ts = pd.date_range(sim["start_date"], periods=n_keep, freq=f"{dt}min")
    pid = np.repeat(ehr.patient_id.to_numpy(), n_keep)
    # The wearable stream is keyed by device, not patient, as a real deployment would be;
    # the study enrollment log is the only link between the two (see linker.py).
    dev_codes = np.random.default_rng([seed, 6]).choice(16 ** 6, n_patients, replace=False)
    device_ids = np.array([f"DEV-{c:06X}" for c in dev_codes])
    enrollment = pd.DataFrame({"patient_id": ehr.patient_id, "device_id": device_ids,
                               "enrolled_on": pd.Timestamp(sim["start_date"]).date(),
                               "devices": "14-day CGM + wrist wearable (simulated)"})
    stage_k, worn_k = stage[:, keep], worn[:, keep]
    stage_str = np.array([SLEEP_STAGES[c] for c in range(-1, 4)], dtype=object)[stage_k + 1]
    stage_str[~worn_k] = None

    wearables = pd.DataFrame({
        "device_id": np.repeat(device_ids, n_keep),
        "timestamp": np.tile(ts, n_patients),
        "cgm_mgdl": cgm[:, keep].ravel(),
        "steps": np.where(worn_k, steps[:, keep], np.nan).ravel(),
        "heart_rate_bpm": np.where(worn_k, hr[:, keep], np.nan).ravel(),
        "sleep_stage": stage_str.ravel(),
    })
    truth = pd.DataFrame({
        "patient_id": pid,
        "timestamp": np.tile(ts, n_patients),
        "glucose_true_mgdl": g_true[:, keep].round(1).ravel(),
        "carbs_g": carbs[:, :, keep].sum(0).round(1).ravel(),
        "insulin_units": doses[:3, :, keep].sum(0).round(2).ravel(),
        "sulfonylurea_mg": doses[3, :, keep].ravel(),
    })
    ev = []
    for s, p_id in zip(schedules, ehr.patient_id):
        for e in s.events:
            if e["step"] >= keep.start:
                ev.append({"patient_id": p_id, "timestamp": ts[e["step"] - keep.start], **{k: e[k] for k in ("event_type", "detail", "value")}})
    events = pd.DataFrame(ev).sort_values(["patient_id", "timestamp"], kind="stable").reset_index(drop=True)

    return {"ehr": ehr, "enrollment": enrollment, "patient_params": params.merge(latent, on="patient_id"),
            "wearables": wearables, "sim_truth": truth, "events": events}


# --------------------------------------------------------------------------- reporting
def summarize(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    """Event rates and sanity statistics used to calibrate and document the simulator."""
    ehr, w = tables["ehr"], tables["wearables"]
    n_p = len(ehr)
    # rows are generated in EHR order; verify rather than assume
    assert (w.device_id.to_numpy()[:: len(w) // n_p] == tables["enrollment"].device_id.to_numpy()).all()
    cgm = w.cgm_mgdl.to_numpy().reshape(n_p, -1)
    ts = pd.DatetimeIndex(w.timestamp.iloc[: cgm.shape[1]])
    labels = label_windows(cgm, ts, cfg)
    labels["regimen"] = ehr.regimen.to_numpy()[labels.patient_idx]
    hypo_drug = (ehr.on_insulin | ehr.on_sulfonylurea).to_numpy()[labels.patient_idx]

    per_pat = pd.DataFrame({
        "mean": np.nanmean(cgm, 1),
        "tir": np.nanmean((cgm >= 70) & (cgm <= 180), 1) / np.mean(~np.isnan(cgm), 1),
        "tbr": np.nanmean(cgm < 70, 1) / np.mean(~np.isnan(cgm), 1),
        "tar": np.nanmean(cgm > 180, 1) / np.mean(~np.isnan(cgm), 1),
    })
    gmi = 3.31 + 0.02392 * per_pat["mean"]
    day_steps = w.groupby([w.device_id, w.timestamp.dt.date]).steps.sum(min_count=1)
    # sleep per recorded night (noon-to-noon), excluding nights the watch was not worn
    night = (w.timestamp - pd.Timedelta(hours=12)).dt.date
    asleep = w.sleep_stage.notna() & (w.sleep_stage != "awake")
    per_night = asleep.groupby([w.device_id, night]).sum() * cfg["sampling"]["interval_minutes"] / 60
    first, last = night.min(), night.max()                        # partial nights at the data edges
    nights = per_night.index.get_level_values(1)
    sleep_h = per_night[(per_night > 0) & (nights != first) & (nights != last)]

    by_reg = labels.groupby("regimen").agg(n_windows=("hypo", "size"), hypo_rate=("hypo", "mean"), hyper_rate=("hyper", "mean"))
    pat_hypo = labels.groupby("patient_idx").hypo.any()
    return {
        "n_patients": int(n_p),
        "n_windows": int(len(labels)),
        "hyper_rate": float(labels.hyper.mean()),
        "hypo_rate": float(labels.hypo.mean()),
        "nocturnal_hypo_rate": float(labels.loc[labels.nocturnal, "hypo"].mean()),
        "nocturnal_window_share": float(labels.nocturnal.mean()),
        "hypo_rate_on_insulin_or_su": float(labels.hypo[hypo_drug].mean()),
        "hypo_rate_other": float(labels.hypo[~hypo_drug].mean()),
        "patients_with_any_hypo_window": float(pat_hypo.mean()),
        "by_regimen": by_reg.round(4).reset_index().to_dict(orient="records"),
        "regimen_counts": ehr.regimen.value_counts().to_dict(),
        "cgm_missing_frac": float(np.isnan(cgm).mean()),
        "mean_glucose_mgdl": float(per_pat["mean"].mean()),
        "tir_mean": float(per_pat.tir.mean()),
        "tbr_mean": float(per_pat.tbr.mean()),
        "tar_mean": float(per_pat.tar.mean()),
        "corr_hba1c_vs_mean_glucose": float(np.corrcoef(ehr.hba1c, per_pat["mean"])[0, 1]),
        "gmi_mean": float(gmi.mean()),
        "hba1c_mean": float(ehr.hba1c.mean()),
        "daily_steps_median": float(day_steps.median()),
        "sleep_hours_mean": float(sleep_h.mean()),
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    s = summary
    lines = [
        "# Synthetic data report",
        "",
        "Auto-generated by `python -m src.data.synthetic_gen`. **All data is synthetic; the EHR-to-sensor",
        "linkage is created by the simulator by design and does not reflect real people.**",
        "",
        f"- Patients: {s['n_patients']}, labelled prediction windows: {s['n_windows']:,}",
        f"- CGM missing (dropouts): {s['cgm_missing_frac']:.1%}",
        "",
        "## Window-level event rates (6 h history -> 2 h horizon, 30 min stride)",
        "",
        "| Target | Rate | Target range |",
        "|---|---|---|",
        f"| Hyperglycaemia (> 180 mg/dL) | {s['hyper_rate']:.1%} | 20-40% |",
        f"| Hypoglycaemia (< 70 mg/dL) | {s['hypo_rate']:.1%} | 3-8% |",
        f"| Nocturnal hypoglycaemia (prediction 22:00-06:00) | {s['nocturnal_hypo_rate']:.1%} | - |",
        "",
        f"Hypo rate on insulin or sulfonylurea: {s['hypo_rate_on_insulin_or_su']:.1%}; "
        f"on metformin only / no drugs: {s['hypo_rate_other']:.2%}.",
        f"Patients with at least one hypo window: {s['patients_with_any_hypo_window']:.0%}.",
        "",
        "| Regimen | Windows | Hypo rate | Hyper rate |",
        "|---|---|---|---|",
        *[f"| {r['regimen']} | {r['n_windows']:,} | {r['hypo_rate']:.2%} | {r['hyper_rate']:.1%} |" for r in s["by_regimen"]],
        "",
        "## Sanity statistics",
        "",
        f"- Mean CGM glucose {s['mean_glucose_mgdl']:.0f} mg/dL; TIR {s['tir_mean']:.0%}, TAR {s['tar_mean']:.0%}, TBR {s['tbr_mean']:.1%}",
        f"- Correlation of EHR HbA1c with simulated mean glucose: {s['corr_hba1c_vs_mean_glucose']:.2f}",
        f"- Mean EHR HbA1c {s['hba1c_mean']:.1f}% vs mean CGM-derived GMI {s['gmi_mean']:.1f}%",
        f"- Median daily steps {s['daily_steps_median']:,.0f}; mean sleep {s['sleep_hours_mean']:.1f} h/night",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_examples(tables: dict[str, pd.DataFrame], path: Path, n_days: int = 3) -> None:
    """Two days-long traces for contrasting patients, with meals and doses marked, for eyeballing realism."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ehr, w, truth, ev = tables["ehr"], tables["wearables"], tables["sim_truth"], tables["events"]
    picks = [ehr[ehr.regimen == "metformin"].patient_id.iloc[0],
             ehr[ehr.on_sulfonylurea & ~ehr.on_insulin].patient_id.iloc[0],
             ehr[ehr.insulin_regimen == "premix_bid"].patient_id.iloc[0]]
    end = w.timestamp.min() + pd.Timedelta(days=n_days)
    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 3.2 * len(picks)), sharex=True)
    device_of = tables["enrollment"].set_index("patient_id").device_id
    for ax, pid in zip(axes, picks):
        m = (w.device_id == device_of[pid]) & (w.timestamp < end)
        ax.axhspan(180, 400, color="#f4d6c6", alpha=0.5, lw=0)
        ax.axhspan(40, 70, color="#c9d7ee", alpha=0.6, lw=0)
        ax.plot(truth.timestamp[m], truth.glucose_true_mgdl[m], color="#999999", lw=0.8, label="true glucose (hidden)")
        ax.plot(w.timestamp[m], w.cgm_mgdl[m], color="#1f4e79", lw=1.2, label="observed CGM")
        e = ev[(ev.patient_id == pid) & (ev.timestamp < end)]
        for _, r in e[e.event_type == "meal"].iterrows():
            ax.annotate(r.detail.split("_")[0], (r.timestamp, 45), fontsize=6, rotation=90, color="#8a5a00")
        for _, r in e[e.event_type.str.contains("dose")].iterrows():
            ax.axvline(r.timestamp, color="#b03a2e", lw=0.6, ls=":")
        asleep = w.sleep_stage[m].notna()
        ax.fill_between(w.timestamp[m], 40, 400, where=asleep, color="#e8e8f4", alpha=0.6, lw=0, step="mid")
        row = ehr.set_index("patient_id").loc[pid]
        ax.set_title(f"{pid}  |  {row.regimen}  |  HbA1c {row.hba1c}%  eGFR {row.egfr:.0f}", fontsize=9, loc="left")
        ax.set_ylim(40, 350)
        ax.set_ylabel("mg/dL")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Synthetic examples (shaded: sleep; dotted red: drug doses). SYNTHETIC DATA.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-patients", type=int, default=None)
    ap.add_argument("--n-days", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    t0 = time.time()
    tables = generate_dataset(cfg, args.n_patients, args.n_days)
    raw = resolve_path(cfg["paths"]["raw_dir"])
    tables["ehr"].to_csv(raw / "ehr.csv", index=False)
    tables["enrollment"].to_csv(raw / "enrollment.csv", index=False)
    tables["patient_params"].to_csv(raw / "patient_params_HIDDEN.csv", index=False)
    tables["events"].to_csv(raw / "events_truth.csv", index=False)
    tables["wearables"].to_csv(raw / "wearables.csv.gz", index=False)
    tables["sim_truth"].to_csv(raw / "sim_truth.csv.gz", index=False)

    summary = summarize(tables, cfg)
    (raw / "sim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    docs = resolve_path(cfg["paths"]["docs_dir"])
    write_report(summary, docs / "synthetic_data_report.md")
    plot_examples(tables, docs / "synthetic_examples.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_regimen"}, indent=2))
    print(pd.DataFrame(summary["by_regimen"]).to_string(index=False))
    print(f"done in {time.time() - t0:.1f}s -> {raw}")


if __name__ == "__main__":
    main()
