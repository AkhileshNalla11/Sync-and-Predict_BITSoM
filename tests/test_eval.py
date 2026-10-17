"""Stage 5 tests: paired comparisons, calibration, event onsets and lead time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import (TAIL_QUANTILES, auprc, calibration_summary, calibration_table, event_onsets,
                              lead_time_stats, paired_bootstrap_diff)


def _toy(n_pat: int = 30, per: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_pat), per)
    y = (rng.random(n_pat * per) < 0.1).astype(int)
    return y, y + rng.normal(0, 1.0, len(y)), groups


# ------------------------------------------------------------------ paired comparisons
def test_paired_diff_of_identical_models_is_zero() -> None:
    y, s, g = _toy()
    d = paired_bootstrap_diff(y, s, s.copy(), g, auprc, 100, 0)
    assert d["diff"] == 0 and d["diff_lo"] == 0 and d["diff_hi"] == 0 and d["p_a_better"] == 0


def test_paired_diff_detects_better_model() -> None:
    y, s, g = _toy()
    better = s + 2.0 * y                                   # strictly more separable
    d = paired_bootstrap_diff(y, better, s, g, auprc, 100, 0)
    assert d["diff"] > 0 and d["diff_lo"] > 0 and d["p_a_better"] == 1.0


# ------------------------------------------------------------------ calibration
def test_calibrated_beats_overconfident() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.3, 50000)
    y = (rng.random(50000) < p).astype(int)
    good = calibration_summary(y, p)
    bad = calibration_summary(y, np.clip(p * 2.5, 0, 1))
    assert good["ece"] < 0.01 < bad["ece"]
    assert good["brier"] < bad["brier"]
    assert good["mean_predicted"] == pytest.approx(y.mean(), abs=0.01)


def test_tail_bins_resolve_rare_events() -> None:
    rng = np.random.default_rng(2)
    p = np.where(rng.random(20000) < 0.95, 0.001, rng.uniform(0.2, 0.9, 20000))
    y = (rng.random(20000) < p).astype(int)
    assert len(calibration_table(y, p)) < len(calibration_table(y, p, TAIL_QUANTILES))
    t = calibration_table(y, p, TAIL_QUANTILES)
    assert t.n.sum() == 20000 and t.mean_predicted.is_monotonic_increasing


# ------------------------------------------------------------------ onsets + lead time
def test_event_onsets_require_clear_period() -> None:
    x = np.full(100, 120.0)
    x[20:25] = 60            # episode 1 onset at 20
    x[27:29] = 65            # re-crossing after only 2 clear samples -> same episode
    x[50:52] = 60            # clear for 21 samples -> new onset at 50
    x[60] = np.nan           # missing reading carries 120 forward
    x[61:63] = 55            # new onset at 61 (clear since 52)
    assert event_onsets(x[None, :], 70, below=True, clear_samples=6)[0].tolist() == [20, 50, 61]
    assert event_onsets(x[None, :], 180, below=False, clear_samples=6)[0].tolist() == []


def test_lead_time_from_earliest_alert() -> None:
    onsets = [np.array([100, 300])]
    anchors = np.arange(0, 400, 6)
    w = pd.DataFrame({"patient_idx": 0, "anchor_idx": anchors, "in_event_hypo": False})
    scores = np.zeros(len(anchors))
    scores[np.isin(anchors, [84, 90, 96])] = 1.0           # alerts 16, 10 and 4 samples before onset 100
    r = lead_time_stats(onsets, w, scores, 0.5, "in_event_hypo", horizon=24, dt_min=5)
    assert r["n_events"] == 2 and r["detection_rate"] == 0.5
    assert r["median_lead_min"] == 80                       # earliest alert: (100 - 84) * 5


def test_lead_time_ignores_alerts_outside_horizon_and_in_event() -> None:
    onsets = [np.array([100])]
    w = pd.DataFrame({"patient_idx": 0, "anchor_idx": [60, 90, 96], "in_event_hypo": [False, True, False]})
    scores = np.array([1.0, 1.0, 0.0])                     # 60 is > 24 samples early; 90 is already in an event
    r = lead_time_stats(onsets, w, scores, 0.5, "in_event_hypo", horizon=24, dt_min=5)
    assert r["n_events"] == 1 and r["detection_rate"] == 0.0 and np.isnan(r["median_lead_min"])
