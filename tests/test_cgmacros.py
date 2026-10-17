"""CGMacros loader tests on a tiny synthetic fixture written in the CGMacros file format (no real data)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.cgmacros import (build_windows, glycaemic_status, load_bio, load_cohort, mask_bridged_gaps,
                               resample_5min)
from src.models.baselines import FEATURE_SETS

START = pd.Timestamp("2020-05-01 10:32:00")       # deliberately off the 5-min clock grid
N_MIN = 2 * 24 * 60                                # two days of 1-min rows
SPIKE_AT = pd.Timestamp("2020-05-02 13:00:00")
GAP = (pd.Timestamp("2020-05-01 20:00:00"), pd.Timestamp("2020-05-01 22:00:00"))      # rows missing
BRIDGE = (pd.Timestamp("2020-05-02 03:00:00"), pd.Timestamp("2020-05-02 04:00:00"))   # straight line
CSV_COLUMNS = ["Unnamed: 0", "Timestamp", "Libre GL", "Dexcom GL", "HR", "Calories (Activity)", "METs",
               "Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber", "Amount Consumed ", "Image path"]


def _dexcom_1min() -> pd.Series:
    """5-min 'native' readings with a smooth daily pattern and one spike, linearly interpolated to 1 min."""
    ts1 = pd.date_range(START, periods=N_MIN, freq="1min")
    native = ts1[::5]
    h = (native - START).total_seconds().to_numpy() / 3600
    g = 120 + 25 * np.sin(2 * np.pi * h / 3.7) + 90 * np.exp(-0.5 * ((native - SPIKE_AT).total_seconds() / 1200) ** 2)
    v = np.interp((ts1 - START).total_seconds(), (native - START).total_seconds(), np.round(g))
    s = pd.Series(v, index=ts1)
    b = s.index.slice_indexer(*BRIDGE)
    s.iloc[b] = np.linspace(s[BRIDGE[0]], s[BRIDGE[1]], b.stop - b.start)
    return s


def _write_participant(root: Path, subject: int, dexcom: pd.Series) -> None:
    d = root / f"CGMacros-{subject:03d}"
    d.mkdir(parents=True)
    df = pd.DataFrame({c: np.nan for c in CSV_COLUMNS}, index=range(len(dexcom)))
    df["Unnamed: 0"] = np.arange(len(dexcom))
    df["Timestamp"] = dexcom.index.strftime("%Y-%m-%d %H:%M:%S")
    df["Dexcom GL"] = dexcom.to_numpy()
    df["Libre GL"] = dexcom.to_numpy() - 30           # different sensor, must not be picked by default
    df["HR"] = 70.0
    df = df[~((dexcom.index >= GAP[0]) & (dexcom.index < GAP[1]))]
    df.loc[df.index[100], "Dexcom GL"] = 999.0        # out of the 40-400 sensor range
    df = pd.concat([df, df.iloc[[50]]])               # duplicated timestamp row
    df.to_csv(d / f"CGMacros-{subject:03d}.csv", index=False)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "CGMacros"
    root.mkdir()
    pd.DataFrame({"subject": [1, 2, 3, 4], "Age": [55, 40, 30, 60], "Gender": ["F", "M", "F", "M"],
                  "BMI": [31.0, 27.5, 22.0, 29.0], "Body weight ": [180, 170, 130, 190],
                  "A1c PDL (Lab)": [7.2, 5.9, 5.2, 6.8]}).to_csv(root / "bio.csv", index=False)
    for subject in (1, 2, 3):                         # subject 4 has no CGM file
        _write_participant(root, subject, _dexcom_1min())
    return tmp_path


@pytest.mark.parametrize("a1c, status", [(5.69, "healthy"), (5.7, "prediabetes"), (6.4, "prediabetes"),
                                         (6.5, "t2d"), (np.nan, "unknown")])
def test_glycaemic_status_cutoffs(a1c: float, status: str) -> None:
    assert glycaemic_status(a1c) == status


def test_load_bio(data_dir: Path) -> None:
    bio = load_bio(data_dir / "CGMacros")
    assert list(bio.index) == ["CGMacros-001", "CGMacros-002", "CGMacros-003", "CGMacros-004"]
    assert list(bio.status) == ["t2d", "prediabetes", "healthy", "t2d"]


def test_cohort_filters_status_and_skips_missing_files(data_dir: Path) -> None:
    assert [p.participant_id for p in load_cohort(data_dir, statuses=("t2d",))] == ["CGMacros-001"]
    assert {p.status for p in load_cohort(data_dir)} == {"t2d", "prediabetes", "healthy"}


def test_series_is_dexcom_on_5min_clock_grid_with_gaps(data_dir: Path) -> None:
    p = load_cohort(data_dir, statuses=("t2d",))[0]
    s = p.cgm
    assert s.index[0] == pd.Timestamp("2020-05-01 10:35:00")
    assert (np.diff(s.index.asi8) == 5 * 60 * 10**9).all()
    truth = _dexcom_1min()
    clean = s.index[(s.index < GAP[0]) & s.notna()][1:5]
    np.testing.assert_allclose(s[clean].to_numpy(), truth[clean].to_numpy())       # Dexcom, not Libre
    assert s[GAP[0]:GAP[1] - pd.Timedelta(minutes=1)].isna().all()               # missing rows -> NaN
    inner = s[BRIDGE[0] + pd.Timedelta(minutes=1):BRIDGE[1] - pd.Timedelta(minutes=1)]
    assert len(inner) >= 10 and inner.isna().all()                                 # straight line masked
    assert p.report["out_of_range_readings"] == 1
    assert p.report["bridged_minutes_masked"] >= 55
    assert p.report["n_gaps_ge_30min"] >= 2 and p.report["longest_gap_h"] >= 2
    assert 0.8 < p.report["coverage"] < 1.0


def test_mask_bridged_gaps_keeps_short_straight_runs() -> None:
    ts = pd.date_range("2020-01-01", periods=40, freq="1min")
    s = pd.Series(np.interp(np.arange(40), np.arange(0, 40, 5), [100, 104, 99, 110, 120, 118, 117, 125]), index=ts)
    masked, n = mask_bridged_gaps(s, 30)
    assert n == 0 and masked.notna().all()
    assert resample_5min(masked).index.minute.isin(range(0, 60, 5)).all()


def test_build_windows_labels_and_features(data_dir: Path, cfg: dict) -> None:
    cohort = load_cohort(data_dir)
    windows, feats = build_windows(cohort, cfg)
    assert len(windows) == len(feats) and set(windows.status) == {"t2d", "prediabetes", "healthy"}
    assert set(FEATURE_SETS["cgm"]) <= set(feats.columns)
    one = windows[windows.patient_id == "CGMacros-001"]
    horizon = pd.Timedelta(hours=cfg["windows"]["horizon_hours"])
    before_spike = one[(one.timestamp < SPIKE_AT) & (one.timestamp >= SPIKE_AT - horizon + pd.Timedelta(minutes=30))]
    assert len(before_spike) > 0 and before_spike.hyper.all()
    far = one[(one.timestamp < SPIKE_AT - pd.Timedelta(hours=4)) | (one.timestamp > SPIKE_AT + pd.Timedelta(hours=2))]
    assert not far.hyper.any() and not one.hypo.any()
    # wearable channels are unobserved, so CGM features must still be defined
    assert feats.cgm_last.notna().all()
