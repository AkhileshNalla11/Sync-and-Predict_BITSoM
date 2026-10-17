"""What-if scenarios: perturb a virtual patient's day, re-simulate, and re-predict both risks.

The twin has two halves, and every result reports both separately:
  * Mechanistic half: the glucose model that generated the data, in one of two modes.
      - "fitted" (default): parameters estimated per patient from the EHR, the first days of CGM and
        the patient's meal / dose log (src/twin/personalise.py); the schedule comes from that log and
        the wearable. A simple state observer pulls the simulation toward the observed CGM up to 'now',
        then it runs freely. Scenario CGM = observed CGM + (simulated scenario - simulated baseline).
        Nothing hidden is used, as in a real deployment.
      - "oracle": replays the simulator's HIDDEN parameters, exact schedule and noise. Only possible
        because the data is synthetic; kept for comparison.
    This yields the *simulated* glucose over the next 2 h under the scenario.
  * ML half: the deployed predictor, re-run on the 6 h history the scenario WOULD have produced,
    plus the edited EHR profile (for medication changes).
Scenario events after the prediction time change only the simulator projection: the ML model can't
see the future, so it cannot react to a dinner that hasn't happened yet. Results flag this explicitly.

Nothing here is clinically valid; the simulator is a simplified synthetic model.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.data.ehr_features import generate_cohort
from src.data.synthetic_gen import (MEAL_CATALOGUE, _Schedule, apply_cgm_noise, derive_patient_params, draw_cgm_noise,
                                    generate_schedule, simulate_glucose, simulate_heart_rate)
from src.twin import personalise
from src.twin.state import TARGETS, PatientTwin, TwinContext

MAIN_MEALS = ("rice_meal", "roti_meal", "idli_dosa", "poha_upma", "paratha")
MODES = ("fitted", "oracle")


# --------------------------------------------------------------------------- exact replay of one patient
@lru_cache(maxsize=2)
def _cohort(seed: int, n_patients: int, total_days: int, dt: int, cal_json: str) -> dict[str, Any]:
    """Regenerate cohort-level parameters and noise exactly as synthetic_gen.generate_dataset drew them."""
    cal = json.loads(cal_json)
    ehr, latent = generate_cohort(n_patients, np.random.default_rng([seed, 0]))
    params = derive_patient_params(ehr, latent, cal, dt, np.random.default_rng([seed, 1]))
    n_t = total_days * 24 * 60 // dt
    return {
        "ehr": ehr, "params": params,
        "g_noise": np.random.default_rng([seed, 3]).normal(0, cal["process_noise_sd"], (n_t, n_patients)),
        "cgm_noise": draw_cgm_noise(n_patients, n_t, dt, np.random.default_rng([seed, 4])),
        "hr_eps": np.random.default_rng([seed, 5]).normal(0, 2.0, (n_t, n_patients)),
    }


@dataclass
class SimRun:
    series: np.ndarray          # (T_keep, 4): cgm, steps, hr, sleep_code, as in data/processed/series.npz
    glucose: np.ndarray         # (T_keep,) simulated glucose (oracle mode: the hidden true glucose)


class PatientSimulator:
    """One patient's mechanistic twin; `run(schedule, anchor_idx)` re-simulates an edited schedule.

    mode="fitted": personalised parameters + schedule from the patient's log and wearable (nothing hidden).
    mode="oracle": exact replay of the hidden simulation (a synthetic-data privilege, kept for comparison).
    """

    def __init__(self, cfg: dict[str, Any], patient_id: str, mode: str = "fitted"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        sim = cfg["simulation"]
        self.mode = mode
        self.dt = cfg["sampling"]["interval_minutes"]
        self.seed = cfg["seed"]
        self.cal = sim["calibration"]
        total_days = sim["n_days"] + sim["warmup_days"]
        self.warm = sim["warmup_days"] * 24 * 60 // self.dt
        self.patient_id = patient_id
        self.fit: personalise.FitResult | None = None
        self.observer_gain = 0.0
        self._anchored: dict[int, tuple[SimRun, np.ndarray, np.ndarray]] = {}
        if mode == "oracle":
            c = _cohort(self.seed, sim["n_patients"], total_days, self.dt, json.dumps(self.cal, sort_keys=True))
            ids = c["ehr"].patient_id.tolist()
            self.i = ids.index(patient_id)
            self.params = c["params"].iloc[[self.i]].reset_index(drop=True)
            self.ehr_row = c["ehr"].iloc[self.i]
            self.base_schedule = generate_schedule(c["params"].iloc[self.i], self.ehr_row, total_days, self.dt,
                                                   np.random.default_rng([self.seed, 2, self.i]))
            i = [self.i]
            self._g_noise, self._hr_eps = c["g_noise"][:, i], c["hr_eps"][:, i]
            n = c["cgm_noise"]
            self._cgm_noise = {"e": n["e"][:, i], "gain": n["gain"][i], "white": n["white"][i], "mask": n["mask"][i]}
            self.baseline = self.run(self.base_schedule)
        else:
            inp = personalise.patient_inputs(cfg, patient_id)
            self.fit = personalise.get_fit(cfg, patient_id)
            self.params = self.fit.params
            self.ehr_row = inp.ehr_row
            self.observed = inp.series
            self.observer_gain = personalise.settings(cfg, len(inp.series) * self.dt // (24 * 60))["observer_gain"]
            self.base_schedule = personalise.schedule_from_log(inp, self.dt, self.warm)
            n_t = self.base_schedule.n
            self._g_noise, self._hr_eps = np.zeros((n_t, 1)), np.zeros((n_t, 1))     # deterministic twin
            self._open_loop_g = self._glucose(self.base_schedule, self._g_noise)
            self.baseline = SimRun(series=self._series(self.base_schedule, self._open_loop_g, self._open_loop_g),
                                   glucose=self._open_loop_g[0, self.warm:])

    # ------------------------------------------------------------------ simulation
    def _glucose(self, s: _Schedule, noise: np.ndarray, obs: np.ndarray | None = None,
                 corrections: np.ndarray | None = None) -> np.ndarray:
        tod = (np.arange(s.n) * self.dt / 60) % 24
        return simulate_glucose(self.params, s.carbs[:, None, :], s.doses[:, None, :], s.steps[None], s.ir_mult[None],
                                tod, self.dt, self.cal, None, noise=noise, obs=obs, obs_gain=self.observer_gain,
                                corrections=corrections)

    def _hr(self, s: _Schedule, g: np.ndarray) -> np.ndarray:
        return simulate_heart_rate(self.params, s.steps[None], s.stage[None], s.ir_mult[None], g, self.dt, None,
                                   eps=self._hr_eps)

    def _series(self, s: _Schedule, g: np.ndarray, g_ref: np.ndarray) -> np.ndarray:
        """Observable channels. Fitted mode: the observed record plus the simulated change relative to `g_ref`."""
        k = slice(self.warm, s.n)
        worn = s.worn[k]
        stage = s.stage[k].astype(np.float32)
        if self.mode == "oracle":
            cgm = apply_cgm_noise(g, self._cgm_noise, self.dt)[0, k]
            hr = self._hr(s, g)[0, k]
        else:
            d_cgm = (personalise.sensor_view(g, self.dt) - personalise.sensor_view(g_ref, self.dt))[0, k]
            cgm = np.clip(np.round(self.observed[:, 0] + d_cgm), 40, 400)
            hr = self.observed[:, 2] + (self._hr(s, g) - self._hr(self.base_schedule, g_ref))[0, k]
        return np.stack([cgm, np.where(worn, s.steps[k], np.nan), np.where(worn, hr, np.nan),
                         np.where(worn & (stage >= 0), stage, np.nan)], axis=1).astype(np.float32)

    def _anchored_baseline(self, anchor_idx: int) -> tuple[SimRun, np.ndarray, np.ndarray]:
        """Fitted mode: baseline with the observer on up to `anchor_idx` -> (run, observer corrections, full glucose)."""
        if anchor_idx not in self._anchored:
            s = self.base_schedule
            obs = personalise.observer_target(self.observed[:, 0], self.warm, s.n, anchor_idx, self.dt)[None]
            corr = np.zeros((s.n, 1))
            g = self._glucose(s, self._g_noise, obs=obs, corrections=corr)
            self._anchored[anchor_idx] = (SimRun(series=self._series(s, g, g), glucose=g[0, self.warm:]), corr, g)
            if len(self._anchored) > 16:
                self._anchored.pop(next(iter(self._anchored)))
        return self._anchored[anchor_idx]

    def baseline_at(self, anchor_idx: int) -> SimRun:
        """The 'as it happened' run that a what-if at this prediction time is compared against."""
        return self.baseline if self.mode == "oracle" else self._anchored_baseline(anchor_idx)[0]

    def run(self, s: _Schedule, anchor_idx: int | None = None) -> SimRun:
        """Simulate a schedule. In fitted mode with an anchor, the observer corrections of that anchor's baseline
        are replayed, so an unchanged schedule reproduces the baseline exactly; without one it runs open loop."""
        if self.mode == "oracle":
            g = self._glucose(s, self._g_noise)
            g_ref = g
        elif anchor_idx is None:
            g, g_ref = self._glucose(s, self._g_noise), self._open_loop_g
        else:
            _, corr, g_ref = self._anchored_baseline(anchor_idx)
            g = self._glucose(s, corr)
        return SimRun(series=self._series(s, g, g_ref), glucose=g[0, self.warm:])

    def events(self, anchor_idx: int, before_h: float = 6, after_h: float = 2) -> pd.DataFrame:
        """Simulator events around a prediction time (absolute steps include the warm-up day)."""
        now = anchor_idx + self.warm
        lo, hi = now - int(before_h * 60 / self.dt), now + int(after_h * 60 / self.dt)
        ev = pd.DataFrame(self.base_schedule.events)
        ev = ev[(ev.step >= lo) & (ev.step <= hi)].copy()
        ev["hours_from_now"] = (ev.step - now) * self.dt / 60
        return ev.reset_index(drop=True)


# --------------------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    """Base class. `apply` edits the schedule (and profile) in place; times are relative to 'now'."""

    def apply(self, s: _Schedule, now: int, profile: dict[str, Any], dt: int) -> list[int]:
        """Apply; return the absolute steps the change touches (used to tell past from future)."""
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


def _step(now: int, hours: float, dt: int) -> int:
    return now + int(round(hours * 60 / dt))


def _find_meal(s: _Schedule, step: int) -> dict[str, Any]:
    for e in s.events:
        if e["event_type"] == "meal" and e["step"] == step:
            return e
    raise ValueError(f"no meal at step {step}")


def _remove_meal(s: _Schedule, e: dict[str, Any]) -> tuple[int, float]:
    cls = MEAL_CATALOGUE[e["detail"]][2]
    s.carbs[cls, e["step"]] = max(0.0, s.carbs[cls, e["step"]] - e["value"])
    if s.carbs[cls, e["step"]] < 0.01:
        s.carbs[cls, e["step"]] = 0.0
    return cls, e["value"]


@dataclass
class AddMeal(Scenario):
    meal: str = "rice_meal"
    hours_from_now: float = 0.0
    carbs_g: float | None = None

    def apply(self, s, now, profile, dt):
        lo, hi, cls = MEAL_CATALOGUE[self.meal]
        t = _step(now, self.hours_from_now, dt)
        s.carbs[cls, t] += self.carbs_g if self.carbs_g is not None else (lo + hi) / 2
        return [t]

    def describe(self):
        g = f"{self.carbs_g:.0f} g" if self.carbs_g is not None else "typical portion"
        return f"Add {self.meal.replace('_', ' ')} ({g}) at {self.hours_from_now:+.1f} h"


@dataclass
class SkipMeal(Scenario):
    """Skip a logged meal; medication is still taken as usual (the classic cause of lows)."""
    meal_step: int = 0

    def apply(self, s, now, profile, dt):
        _remove_meal(s, _find_meal(s, self.meal_step))
        return [self.meal_step]

    def describe(self):
        return "Skip a meal (medication taken as usual)"


@dataclass
class DelayMeal(Scenario):
    """Eat a logged meal later (e.g. late dinner); medication timing unchanged."""
    meal_step: int = 0
    delay_hours: float = 2.0

    def apply(self, s, now, profile, dt):
        cls, grams = _remove_meal(s, _find_meal(s, self.meal_step))
        t = min(s.n - 1, self.meal_step + int(round(self.delay_hours * 60 / dt)))
        s.carbs[cls, t] += grams
        return [self.meal_step, t]

    def describe(self):
        return f"Delay a meal by {self.delay_hours:.1f} h (medication at usual time)"


@dataclass
class PoorSleep(Scenario):
    """Turn the most recent night before 'now' into a short, fragmented night.

    Effects follow the simulator: fragmented stages, waking 1.5 h early, and ~20 % higher insulin
    resistance for the following day.
    """
    seed: int = 0

    def apply(self, s, now, profile, dt):
        nights = [e for e in s.events if e["event_type"] == "sleep" and e["step"] <= now]
        if not nights:
            return []
        onset = nights[-1]["step"]
        wake = min(s.n, onset + int(nights[-1]["value"] * 60 / dt))
        new_wake = max(onset + 1, wake - int(1.5 * 60 / dt))
        rng = np.random.default_rng(self.seed)
        night = s.stage[onset:new_wake]
        night[(night == 2) & (rng.random(len(night)) < 0.6)] = 1          # lose deep sleep
        night[rng.random(len(night)) < 0.12] = 0                          # wake bouts
        s.stage[new_wake:wake] = -1                                       # up early
        nxt = [e["step"] for e in s.events if e["event_type"] == "sleep" and e["step"] > onset]
        day_end = nxt[0] if nxt else s.n
        s.ir_mult[new_wake:day_end] = np.maximum(s.ir_mult[new_wake:day_end], 1.2)
        return [onset, new_wake]

    def describe(self):
        return "Poor sleep last night (short, fragmented)"


@dataclass
class AddActivity(Scenario):
    hours_from_now: float = -1.0
    minutes: float = 45
    steps_per_5min: float = 480

    def apply(self, s, now, profile, dt):
        a = _step(now, self.hours_from_now, dt)
        b = min(s.n, a + int(round(self.minutes / dt)))
        s.steps[a:b] = np.maximum(s.steps[a:b], self.steps_per_5min)
        return list(range(a, b))

    def describe(self):
        return f"Brisk walk, {self.minutes:.0f} min at {self.hours_from_now:+.1f} h"


@dataclass
class ExtraDose(Scenario):
    """kind: 'rapid' (insulin), 'premix' (30/70 insulin), 'basal' (insulin) or 'sulfonylurea' (mg)."""
    kind: str = "premix"
    amount: float = 8.0
    hours_from_now: float = -1.0

    def apply(self, s, now, profile, dt):
        t = _step(now, self.hours_from_now, dt)
        if self.kind == "premix":
            s.doses[0, t] += 0.3 * self.amount
            s.doses[1, t] += 0.7 * self.amount
        else:
            s.doses[{"rapid": 0, "basal": 2, "sulfonylurea": 3}[self.kind], t] += self.amount
        return [t]

    def describe(self):
        unit = "mg" if self.kind == "sulfonylurea" else "U"
        return f"Extra {self.kind} dose, {self.amount:g} {unit} at {self.hours_from_now:+.1f} h"


@dataclass
class ScaleMedication(Scenario):
    """Change the prescribed insulin / sulfonylurea dose by a factor for the whole record.

    Updates the EHR dose fields too, so the ML model sees the new prescription.
    """
    factor: float = 1.5

    def apply(self, s, now, profile, dt):
        s.doses *= self.factor
        for col in ("insulin_total_daily_units", "sulfonylurea_dose_mg"):
            profile[col] = float(profile[col]) * self.factor
        return [0]                                       # affects the past as well

    def describe(self):
        return f"Medication dose x{self.factor:g} (insulin and sulfonylurea)"


# --------------------------------------------------------------------------- run
@dataclass
class WhatIfResult:
    baseline: PatientTwin
    scenario: PatientTwin
    projection: pd.DataFrame              # past 6 h + next 2 h: simulated & CGM glucose, baseline vs scenario
    simulated_outcome: dict[str, dict[str, float | bool]]
    descriptions: list[str]
    unseen_by_model: list[str] = field(default_factory=list)   # scenario parts after 'now'
    mode: str = "fitted"                  # which mechanistic twin produced the projection (see PatientSimulator)

    @property
    def risk_change(self) -> dict[str, float]:
        return {t: self.scenario.risk[t] - self.baseline.risk[t] for t in TARGETS}


def _outcome(run: SimRun, anchor: int, horizon: int, thr: dict[str, Any]) -> dict[str, float | bool]:
    fut = slice(anchor + 1, anchor + 1 + horizon)
    cgm = run.series[fut, 0]
    g = run.glucose[fut]
    return {"sim_max": float(g.max()), "sim_min": float(g.min()),
            "hyper": bool(np.nanmax(cgm) > thr["hyper_mgdl"]) if np.isfinite(cgm).any() else False,
            "hypo": bool(np.nanmin(cgm) < thr["hypo_mgdl"]) if np.isfinite(cgm).any() else False}


def run_whatif(ctx: TwinContext, sim: PatientSimulator, anchor_idx: int, scenarios: list[Scenario]) -> WhatIfResult:
    """Apply scenarios to the patient's day at a prediction time; re-simulate and re-predict."""
    dt = ctx.dt
    now = anchor_idx + sim.warm
    sched = copy.deepcopy(sim.base_schedule)
    profile = dict(ctx.ehr.loc[sim.patient_id].to_dict())
    touched: list[tuple[str, list[int]]] = []
    for sc in scenarios:
        touched.append((sc.describe(), sc.apply(sched, now, profile, dt)))
    base = sim.baseline_at(anchor_idx)
    run = sim.run(sched, anchor_idx)

    base_twin = ctx.build_twin(sim.patient_id, anchor_idx)
    scen_twin = ctx.build_twin(sim.patient_id, anchor_idx, series_row=run.series, profile=profile)
    horizon = ctx.cfg["windows"]["horizon_hours"] * 60 // dt
    sl = slice(anchor_idx - ctx.history_len + 1, anchor_idx + 1 + horizon)
    proj = pd.DataFrame({
        "timestamp": ctx.timestamps[sl],
        "phase": np.where(np.arange(sl.start, sl.stop) <= anchor_idx, "history", "next 2 h"),
        "sim_baseline": base.glucose[sl], "sim_scenario": run.glucose[sl],
        "cgm_baseline": base.series[sl, 0], "cgm_scenario": run.series[sl, 0],
    })
    thr = ctx.cfg["thresholds"]
    unseen = [d for d, steps in touched if steps and min(steps) > now]
    return WhatIfResult(
        baseline=base_twin, scenario=scen_twin, projection=proj,
        simulated_outcome={"baseline": _outcome(base, anchor_idx, horizon, thr),
                           "scenario": _outcome(run, anchor_idx, horizon, thr)},
        descriptions=[d for d, _ in touched], unseen_by_model=unseen, mode=sim.mode)
