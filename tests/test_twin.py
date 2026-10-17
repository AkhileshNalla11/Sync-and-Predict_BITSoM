"""Stage 6 tests: twin state, explanations, exact replay and what-if scenarios.

Builds a small self-contained pipeline (24 patients x 3 days) in a temp directory.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.data.build_dataset import build_dataset, save_dataset
from src.data.synthetic_gen import generate_dataset
from src.models.baselines import run as run_baselines
from src.twin.explain import contributions, explain, group_contributions
from src.twin.state import TwinContext
from src.twin.whatif import (AddActivity, AddMeal, DelayMeal, ExtraDose, PatientSimulator, PoorSleep, ScaleMedication,
                             SkipMeal, run_whatif)

N_PAT, N_DAYS = 24, 3


@pytest.fixture(scope="module")
def ctx(tmp_path_factory) -> TwinContext:
    root = tmp_path_factory.mktemp("twin")
    cfg = copy.deepcopy(load_config())
    cfg["paths"] = {k: str(root / k) for k in ("raw_dir", "processed_dir", "docs_dir")}
    cfg["paths"]["models_dir"] = str(root / "processed" / "models")
    cfg["simulation"].update(n_patients=N_PAT, n_days=N_DAYS)
    cfg["eval"]["n_bootstrap"] = 5
    tables = generate_dataset(cfg)
    (root / "raw_dir").mkdir()
    tables["ehr"].to_csv(root / "raw_dir" / "ehr.csv", index=False)
    save_dataset(build_dataset(cfg, tables), cfg)
    run_baselines(cfg, ["cgm_wearables_ehr"])
    return TwinContext.load(cfg)


def _anchor(ctx: TwinContext, pid: str, k: int = -10) -> int:
    return int(ctx.patient_windows(pid).anchor_idx.iloc[k])


def _drug_patient(ctx: TwinContext, col: str = "on_sulfonylurea") -> str:
    cands = [p for p in ctx.patient_ids if bool(ctx.ehr.loc[p, col])]
    if not cands:
        pytest.skip(f"no {col} patient in the small cohort")
    return cands[0]


# ------------------------------------------------------------------ state
def test_twin_matches_timeline(ctx: TwinContext) -> None:
    pid = ctx.patient_ids[0]
    tl = ctx.risk_timeline(pid)
    tw = ctx.build_twin(pid, int(tl.anchor_idx.iloc[5]))
    row = tl.iloc[5]
    assert tw.risk["hyper"] == pytest.approx(row.hyper_risk) and tw.risk["hypo"] == pytest.approx(row.hypo_risk)
    assert tw.alert["hypo"] == (tw.risk["hypo"] >= ctx.thresholds["hypo"])
    assert len(tw.history) == 72 and tw.history.timestamp.iloc[-1] == tw.time
    assert tw.outcome == {"hyper": bool(row.hyper), "hypo": bool(row.hypo)}


def test_twin_at_time_picks_latest_anchor(ctx: TwinContext) -> None:
    pid = ctx.patient_ids[1]
    w = ctx.patient_windows(pid)
    target = ctx.timestamps[int(w.anchor_idx.iloc[20])] + np.timedelta64(10, "m")
    assert ctx.twin(pid, at=target).anchor_idx == int(w.anchor_idx.iloc[20])


def test_alert_thresholds_from_validation(ctx: TwinContext) -> None:
    path = next(Path(ctx.cfg["paths"]["models_dir"]).glob("*alert_thresholds.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["false_alert_rate"] == ctx.cfg["twin"]["alert_false_alert_rate"]
    assert 0 < ctx.thresholds["hyper"] < 1 and 0 < ctx.thresholds["hypo"] < 1


# ------------------------------------------------------------------ explanations
def test_native_treeshap_matches_shap_library(ctx: TwinContext) -> None:
    shap = pytest.importorskip("shap")
    tw = ctx.build_twin(ctx.patient_ids[2], _anchor(ctx, ctx.patient_ids[2]))
    for t in ("hyper", "hypo"):
        ours, base = contributions(ctx, tw.features, t)
        theirs = shap.TreeExplainer(ctx.boosters[t]).shap_values(tw.features)
        theirs = theirs[1] if isinstance(theirs, list) else theirs
        np.testing.assert_allclose(ours, np.asarray(theirs).reshape(ours.shape), atol=1e-6)


def test_explanations_are_additive_and_sorted(ctx: TwinContext) -> None:
    tw = ctx.build_twin(ctx.patient_ids[3], _anchor(ctx, ctx.patient_ids[3]))
    for t in ("hyper", "hypo"):
        sv, base = contributions(ctx, tw.features, t)
        assert 1 / (1 + np.exp(-(sv.sum() + base))) == pytest.approx(tw.risk[t], rel=1e-6)
        d = explain(ctx, tw, t, top_k=8)
        assert d.contribution.abs().is_monotonic_decreasing and len(d) == 8
        assert not d.feature.isin(["tod_sin", "tod_cos", "hour"]).any()          # merged into time_of_day
        assert group_contributions(ctx, tw, t).sum() == pytest.approx(sv.sum())


# ------------------------------------------------------------------ simulator replay + scenarios
def test_replay_reproduces_stored_series_exactly(ctx: TwinContext) -> None:
    for pid in ctx.patient_ids[:3]:
        sim = PatientSimulator(ctx.cfg, pid)
        assert np.array_equal(sim.baseline.series, ctx.series[ctx._pidx[pid]], equal_nan=True)


def test_empty_scenario_changes_nothing(ctx: TwinContext) -> None:
    pid = ctx.patient_ids[0]
    sim = PatientSimulator(ctx.cfg, pid)
    r = run_whatif(ctx, sim, _anchor(ctx, pid), [])
    assert r.risk_change == {"hyper": 0.0, "hypo": 0.0}
    np.testing.assert_array_equal(r.projection.true_baseline, r.projection.true_scenario)


def test_future_meal_is_invisible_to_model_but_not_to_simulator(ctx: TwinContext) -> None:
    pid = ctx.patient_ids[4]
    sim = PatientSimulator(ctx.cfg, pid)
    carbs_before = sim.base_schedule.carbs.sum()
    r = run_whatif(ctx, sim, _anchor(ctx, pid), [AddMeal("rice_meal", hours_from_now=0.5)])
    assert r.risk_change == {"hyper": 0.0, "hypo": 0.0}
    assert r.unseen_by_model and r.simulated_outcome["scenario"]["true_max"] > r.simulated_outcome["baseline"]["true_max"] + 20
    assert sim.base_schedule.carbs.sum() == carbs_before                       # baseline never mutated


def test_past_meal_raises_hyper_risk(ctx: TwinContext) -> None:
    pid = ctx.patient_ids[5]
    sim = PatientSimulator(ctx.cfg, pid)
    r = run_whatif(ctx, sim, _anchor(ctx, pid), [AddMeal("sweets", hours_from_now=-0.5, carbs_g=60)])
    assert r.scenario.risk["hyper"] > r.baseline.risk["hyper"] and not r.unseen_by_model


def test_extra_dose_and_activity_lower_glucose(ctx: TwinContext) -> None:
    pid = _drug_patient(ctx)
    sim = PatientSimulator(ctx.cfg, pid)
    a = _anchor(ctx, pid)
    for sc in (ExtraDose("sulfonylurea", 2, -2.0), ExtraDose("premix", 10, -1.0), AddActivity(-1.0, 60)):
        r = run_whatif(ctx, sim, a, [sc])
        assert r.simulated_outcome["scenario"]["true_min"] < r.simulated_outcome["baseline"]["true_min"], sc


def test_skip_and_delay_meal(ctx: TwinContext) -> None:
    pid = ctx.patient_ids[6]
    sim = PatientSimulator(ctx.cfg, pid)
    a = _anchor(ctx, pid, -20)
    meals = sim.events(a, before_h=6, after_h=0)
    meals = meals[meals.event_type == "meal"]
    if meals.empty:
        pytest.skip("no meal in window")
    step, grams = int(meals.step.iloc[-1]), float(meals.value.iloc[-1])
    for sc, expect in ((SkipMeal(step), -grams), (DelayMeal(step, 1.0), 0.0)):
        s = copy.deepcopy(sim.base_schedule)
        sc.apply(s, a + sim.warm, {}, ctx.dt)
        assert s.carbs.sum() - sim.base_schedule.carbs.sum() == pytest.approx(expect, abs=0.02)


def test_scale_medication_updates_profile(ctx: TwinContext) -> None:
    pid = _drug_patient(ctx)
    sim = PatientSimulator(ctx.cfg, pid)
    r = run_whatif(ctx, sim, _anchor(ctx, pid), [ScaleMedication(0.5)])
    assert r.scenario.profile["sulfonylurea_dose_mg"] == pytest.approx(0.5 * float(ctx.ehr.loc[pid, "sulfonylurea_dose_mg"]))
    assert r.simulated_outcome["scenario"]["true_min"] > r.simulated_outcome["baseline"]["true_min"]


def test_poor_sleep_raises_next_day_glucose(ctx: TwinContext) -> None:
    pid = ctx.patient_ids[7]
    sim = PatientSimulator(ctx.cfg, pid)
    w = ctx.patient_windows(pid)
    a = int(w.anchor_idx[(ctx.timestamps[w.anchor_idx].hour == 12)].iloc[-1])   # midday, after a night
    r = run_whatif(ctx, sim, a, [PoorSleep()])
    past = r.projection[r.projection.phase == "history"]
    assert (past.true_scenario >= past.true_baseline - 1e-9).mean() > 0.9
