"""Stage 3 tests: handcrafted features, persistence, metrics, LightGBM baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.build_dataset import build_dataset
from src.data.ehr_features import STATIC_FEATURES
from src.data.synthetic_gen import generate_dataset
from src.data.wearable_features import (FEATURE_GROUPS, NOT_IN_WINDOW_MIN, SERIES_CHANNELS, _slope,
                                        build_feature_table, handcrafted_features, label_windows)
from src.eval.metrics import (evaluate_scores, evaluation_subsets, patient_bootstrap, auroc,
                              sensitivity_at_far, threshold_at_far)
from src.models.baselines import feature_matrix, persistence_scores, train_lgbm

H = 72


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def _window(cgm: np.ndarray, steps: float = 0.0, hr: float = 70.0, sleep: float = np.nan) -> np.ndarray:
    raw = np.empty((1, H, len(SERIES_CHANNELS)), dtype=np.float32)
    raw[0, :, 0], raw[0, :, 1], raw[0, :, 2], raw[0, :, 3] = cgm, steps, hr, sleep
    return raw


# ------------------------------------------------------------------ features
def test_slope_of_linear_ramp() -> None:
    y = 100 + 2.0 * np.arange(13) * 5          # +2 mg/dL per minute
    assert _slope(y[None, :], 5)[0] == pytest.approx(2.0)
    y_gap = y.astype(float)
    y_gap[[3, 7]] = np.nan
    assert _slope(y_gap[None, :], 5)[0] == pytest.approx(2.0)
    assert np.isnan(_slope(np.array([[np.nan, 100.0, np.nan]]), 5)[0])


def test_handcrafted_values(cfg: dict) -> None:
    cgm = np.full(H, 120.0)
    cgm[-6:] = [125, 135, 145, 152, 158, 162]  # meal-like rise over the last 30 min
    f = handcrafted_features(_window(cgm, steps=10.0), np.array([H - 1]), 0.0, 5, cfg).iloc[0]
    assert set(f.index) == {c for g in FEATURE_GROUPS.values() for c in g}
    assert f.cgm_last == 162 and f.cgm_max == 162 and f.cgm_min == 120
    assert f.cgm_slope_30m > 1.0                                   # clearly rising
    assert f.min_since_meal_rise <= 10                              # rise detected at the end
    assert f.min_since_below_70 == NOT_IN_WINDOW_MIN and f.cgm_frac_below_70 == 0
    assert f.steps_1h == 120 and f.cgm_coverage == 1.0
    assert f.hour == pytest.approx(5 + 55 / 60)                     # anchor at sample 71 -> 05:55


def test_minutes_since_low(cfg: dict) -> None:
    cgm = np.full(H, 110.0)
    cgm[H - 13] = 65.0                                               # 12 samples = 60 min ago
    f = handcrafted_features(_window(cgm), np.array([H - 1]), 0.0, 5, cfg).iloc[0]
    assert f.min_since_below_70 == 60 and f.cgm_min == 65


def test_features_use_history_only(cfg: dict) -> None:
    """Changing the future must not change features (no look-ahead)."""
    tables = generate_dataset(cfg, 6, 2)
    ds = build_dataset(cfg, tables)
    w = ds["windows"]
    a = build_feature_table(ds["series"], w, ds["timestamps"], cfg)
    future = ds["series"].copy()
    last_anchor = w.anchor_idx.max()
    future[:, last_anchor + 1:, :] = 999.0
    b = build_feature_table(future, w, ds["timestamps"], cfg)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------ persistence
def test_persistence_scores_follow_trend() -> None:
    feats = pd.DataFrame({"cgm_last": [150.0, 150.0, 90.0], "cgm_slope_30m": [1.0, 0.0, -0.5]})
    s = persistence_scores(feats, 120)
    assert s["hyper"][0] == pytest.approx(270) and s["hyper"][1] == pytest.approx(150)
    assert s["hypo"][2] == pytest.approx(-30)                        # 90 - 60 -> predicted 30
    assert s["hypo"][2] > s["hypo"][1]                               # falling low glucose = more hypo risk


# ------------------------------------------------------------------ metrics
def test_threshold_respects_false_alert_rate() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(100)].astype(int)
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(2, 1, 100)]
    for far in (0.05, 0.10):
        thr = threshold_at_far(y, s, far)
        assert (s[y == 0] > thr).mean() <= far + 1e-12
        assert sensitivity_at_far(y, s, far) == pytest.approx((s[y == 1] > thr).mean())
    assert sensitivity_at_far(y, np.r_[np.zeros(1000), np.ones(100)], 0.05) == 1.0


def test_bootstrap_ci_brackets_estimate(cfg: dict) -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 50)
    y = (rng.random(2000) < 0.2).astype(int)
    s = y + rng.normal(0, 0.8, 2000)
    lo, hi = patient_bootstrap(y, s, groups, auroc, 200, 0)
    assert lo < auroc(y, s) < hi
    res = evaluate_scores(y, s, groups, cfg)
    assert res["n_patients"] == 40 and res["auprc_lo"] <= res["auprc"] <= res["auprc_hi"]


def test_onset_subset_excludes_ongoing_events(cfg: dict) -> None:
    cgm = np.full((1, 400), 120.0)
    cgm[0, 200:230] = 60.0                    # an ongoing low
    w = label_windows(cgm, pd.date_range("2026-01-05", periods=400, freq="5min"), cfg)
    w["patient_id"] = "P"
    in_low = w.anchor_idx.between(200, 229)
    assert w.loc[in_low, "in_event_hypo"].all() and not w.loc[~in_low, "in_event_hypo"].any()
    subsets = {(t, s): m for t, s, m in evaluation_subsets(w)}
    assert not subsets[("hypo", "onset")][in_low.to_numpy()].any()
    assert subsets[("hypo", "all")].all()


# ------------------------------------------------------------------ LightGBM
def test_feature_matrix_static_alignment(cfg: dict) -> None:
    tables = generate_dataset(cfg, 6, 2)
    ds = build_dataset(cfg, tables)
    feats = build_feature_table(ds["series"], ds["windows"], ds["timestamps"], cfg)
    X = feature_matrix(feats, ds["windows"], ds["static"], "cgm_wearables_ehr")
    row = ds["windows"].iloc[123]
    np.testing.assert_allclose(X.iloc[123][list(STATIC_FEATURES)].to_numpy(float),
                               ds["static"].loc[row.patient_id].to_numpy(float))
    assert not any(c in STATIC_FEATURES for c in feature_matrix(feats, ds["windows"], ds["static"], "cgm").columns)


@pytest.mark.slow
def test_lgbm_learns_hyper(cfg: dict) -> None:
    tables = generate_dataset(cfg, 40, 4)
    ds = build_dataset(cfg, tables)
    w = ds["windows"]
    X = feature_matrix(build_feature_table(ds["series"], w, ds["timestamps"], cfg), w, ds["static"], "cgm")
    y = w.hyper.to_numpy(int)
    tr, va, te = (w.split == s for s in ("train", "val", "test"))
    booster = train_lgbm(X[tr.to_numpy()], y[tr], X[va.to_numpy()], y[va], cfg)
    assert auroc(y[te], booster.predict(X[te.to_numpy()])) > 0.8
