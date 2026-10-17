"""Stage 2 tests: window shapes, label correctness, linkage, static encoding, patient-level split."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.build_dataset import (PatientLeakageError, assert_no_patient_leakage, build_dataset,
                                    patient_split)
from src.data.ehr_features import NUMERIC_FEATURES, STATIC_FEATURES, StaticEncoder, encode_static, generate_cohort
from src.data.linker import LinkageError, link_by_attributes, link_by_enrollment
from src.data.synthetic_gen import generate_dataset
from src.data.wearable_features import (SEQUENCE_FEATURES, SERIES_CHANNELS, encode_sequences, extract_windows,
                                        label_windows, window_sizes)

H, F = 72, 24          # 6 h history, 2 h horizon at 5-min sampling


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def stride1_cfg(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["windows"]["stride_minutes"] = c["sampling"]["interval_minutes"]
    return c


@pytest.fixture(scope="module")
def small(cfg: dict) -> dict:
    tables = generate_dataset(cfg, 30, 3)
    return {"tables": tables, "ds": build_dataset(cfg, tables)}


def _ts(n: int, start: str = "2026-01-05 00:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="5min")


# ------------------------------------------------------------------ labels
def test_window_sizes(cfg: dict) -> None:
    assert window_sizes(cfg) == (H, F, 6)


def _labels_at(cgm_row: np.ndarray, cfg: dict) -> pd.DataFrame:
    return label_windows(cgm_row[None, :], _ts(len(cgm_row)), cfg).set_index("anchor_idx")


@pytest.mark.parametrize("value, hyper, hypo", [(181, True, False), (180, False, False),
                                                 (69, False, True), (70, False, False)])
def test_label_thresholds_are_strict(stride1_cfg: dict, value: float, hyper: bool, hypo: bool) -> None:
    cgm = np.full(H + F + 10, 120.0)
    anchor = H - 1
    cgm[anchor + 5] = value
    lab = _labels_at(cgm, stride1_cfg)
    assert lab.loc[anchor, "hyper"] == hyper
    assert lab.loc[anchor, "hypo"] == hypo


def test_label_horizon_boundaries(stride1_cfg: dict) -> None:
    """The horizon is samples t+1 .. t+F: the anchor itself and t+F+1 are excluded."""
    cgm = np.full(H + 2 * F + 40, 120.0)
    event = H + F + 20
    cgm[event] = 250.0
    lab = _labels_at(cgm, stride1_cfg)
    assert not lab.loc[event, "hyper"]                 # value at the prediction time is history, not future
    assert lab.loc[event - 1, "hyper"]                 # first horizon sample
    assert lab.loc[event - F, "hyper"]                 # last horizon sample
    assert not lab.loc[event - F - 1, "hyper"]         # just out of reach


def test_label_uses_both_targets_independently(stride1_cfg: dict) -> None:
    cgm = np.full(H + F + 5, 120.0)
    cgm[H + 2], cgm[H + 10] = 60.0, 200.0
    lab = _labels_at(cgm, stride1_cfg)
    assert lab.loc[H - 1, "hyper"] and lab.loc[H - 1, "hypo"]


def test_low_coverage_windows_dropped(stride1_cfg: dict) -> None:
    cgm = np.full(H + F + 5, 120.0)
    cgm[H: H + F] = np.nan                             # horizon of anchor H-1 fully missing
    lab = _labels_at(cgm, stride1_cfg)
    assert H - 1 not in lab.index
    cgm2 = np.full(H + F + 5, 120.0)
    cgm2[: H // 2 + 1] = np.nan                        # > 50 % of first history missing
    assert H - 1 not in _labels_at(cgm2, stride1_cfg).index


def test_nocturnal_flag(stride1_cfg: dict) -> None:
    n = 288 * 2
    lab = label_windows(np.full((1, n), 120.0), _ts(n), stride1_cfg)
    ts = _ts(n)[lab.anchor_idx]
    noct = pd.Series(lab.nocturnal.to_numpy(), index=ts)
    assert noct[pd.Timestamp("2026-01-05 23:00")]
    assert noct[pd.Timestamp("2026-01-06 05:55")]
    assert not noct[pd.Timestamp("2026-01-06 06:00")]
    assert not noct[pd.Timestamp("2026-01-05 21:55")]
    assert noct[pd.Timestamp("2026-01-05 22:00")]


# ------------------------------------------------------------------ windows
def test_series_and_window_shapes(small: dict, cfg: dict) -> None:
    ds = small["ds"]
    series, w = ds["series"], ds["windows"]
    assert series.shape == (30, 3 * 288, len(SERIES_CHANNELS))
    batch = w.sample(200, random_state=0)
    raw = extract_windows(series, batch.patient_idx.to_numpy(), batch.anchor_idx.to_numpy(), H)
    assert raw.shape == (200, H, len(SERIES_CHANNELS))
    # the last history sample is the series value at the prediction time
    expect = series[batch.patient_idx.to_numpy(), batch.anchor_idx.to_numpy()]
    np.testing.assert_array_equal(raw[:, -1], expect)
    enc = encode_sequences(raw, batch.anchor_idx.to_numpy(), start_hour=0.0, dt_min=5)
    assert enc.shape == (200, H, len(SEQUENCE_FEATURES))
    assert np.isfinite(enc).all()


def test_labels_match_series(small: dict) -> None:
    """Recompute a sample of labels directly from the stored series."""
    ds = small["ds"]
    cgm = ds["series"][..., 0]
    for _, r in ds["windows"].sample(300, random_state=1).iterrows():
        future = cgm[r.patient_idx, r.anchor_idx + 1: r.anchor_idx + 1 + F]
        assert r.hyper == bool(np.nanmax(future) > 180)
        assert r.hypo == bool(np.nanmin(future) < 70)


def test_time_of_day_encoding() -> None:
    raw = np.full((1, H, len(SERIES_CHANNELS)), np.nan, dtype=np.float32)
    anchor = H - 1 + 144                                  # sample 215 -> 17:55 when the series starts at 00:00
    enc = encode_sequences(raw, np.array([anchor]), start_hour=0.0, dt_min=5)
    hour = (np.degrees(np.arctan2(enc[0, -1, -2], enc[0, -1, -1])) % 360) / 15
    assert hour == pytest.approx(17 + 55 / 60, abs=1e-3)


def test_forward_fill_and_masks() -> None:
    raw = np.full((1, H, len(SERIES_CHANNELS)), np.nan, dtype=np.float32)
    raw[0, 10, 0], raw[0, 50, 0] = 100.0, 200.0
    enc = encode_sequences(raw, np.array([H - 1]), 0.0, 5)
    cgm = enc[0, :, SEQUENCE_FEATURES.index("cgm")] * 50 + 140
    obs = enc[0, :, SEQUENCE_FEATURES.index("cgm_observed")]
    assert cgm[0] == pytest.approx(100) and cgm[30] == pytest.approx(100) and cgm[-1] == pytest.approx(200)
    assert obs.sum() == 2


# ------------------------------------------------------------------ linkage
def test_enrollment_linkage(small: dict) -> None:
    t = small["tables"]
    linked, report = link_by_enrollment(t["ehr"], t["wearables"], t["enrollment"])
    assert report.n_linked == 30 and not report.ehr_without_device and not report.devices_without_enrollment
    assert report.synthetic_linkage
    truth = t["enrollment"].set_index("device_id").patient_id
    assert (linked.patient_id.to_numpy() == truth.loc[linked.device_id].to_numpy()).all()


def test_enrollment_rejects_duplicates(small: dict) -> None:
    t = small["tables"]
    bad = t["enrollment"].copy()
    bad.loc[1, "device_id"] = bad.loc[0, "device_id"]
    with pytest.raises(LinkageError):
        link_by_enrollment(t["ehr"], t["wearables"], bad)


def test_orphan_devices_reported(small: dict) -> None:
    t = small["tables"]
    enr = t["enrollment"].iloc[1:]
    linked, report = link_by_enrollment(t["ehr"], t["wearables"], enr)
    assert report.devices_without_enrollment == [t["enrollment"].device_id.iloc[0]]
    assert report.ehr_without_device == [t["enrollment"].patient_id.iloc[0]]
    assert t["enrollment"].device_id.iloc[0] not in set(linked.device_id)


def test_attribute_linkage_one_to_one() -> None:
    ehr = pd.DataFrame({"patient_id": ["A", "B", "C"], "age": [40, 60, 61], "bmi": [24, 30, 22], "sex": ["M", "F", "M"]})
    dev = pd.DataFrame({"device_id": ["d1", "d2", "d3"], "age": [59, 41, 62], "bmi": [29, 25, 23], "sex": ["F", "M", "F"]})
    out = link_by_attributes(ehr, dev, numeric=["age", "bmi"], exact=["sex"])
    pairs = dict(zip(out.device_id, out.patient_id))
    assert pairs == {"d1": "B", "d2": "A"}            # d3 (F) has no unused female partner left
    assert out.patient_id.is_unique and out.method.str.contains("SYNTHETIC").all()


# ------------------------------------------------------------------ static features
def test_static_encoder_fit_on_train_only() -> None:
    ehr, _ = generate_cohort(200, np.random.default_rng(1))
    raw = encode_static(ehr)
    assert list(raw.columns) == list(STATIC_FEATURES) and np.isfinite(raw.to_numpy()).all()
    train = raw.iloc[:150]
    enc = StaticEncoder().fit(train)
    scaled = enc.transform(raw)
    np.testing.assert_allclose(scaled.iloc[:150][list(NUMERIC_FEATURES)].mean(), 0, atol=1e-9)
    restored = StaticEncoder.from_dict(enc.to_dict()).transform(raw)
    pd.testing.assert_frame_equal(scaled, restored)
    assert "patient_id" not in scaled.columns and "diagnoses_icd10" not in scaled.columns


# ------------------------------------------------------------------ patient-level split
def test_split_is_by_patient(cfg: dict) -> None:
    ehr, _ = generate_cohort(300, np.random.default_rng(2))
    s = patient_split(ehr, cfg)
    assert s.patient_id.is_unique and set(s.patient_id) == set(ehr.patient_id)
    sets = {k: set(s.patient_id[s.split == k]) for k in ("train", "val", "test")}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"]) and not (sets["val"] & sets["test"])
    assert abs(len(sets["test"]) / 300 - cfg["split"]["test_fraction"]) < 0.03
    for stratum in ("insulin", "sulfonylurea", "other"):
        assert (s[s.stratum == stratum].split == "test").any()
    pd.testing.assert_frame_equal(s, patient_split(ehr, cfg))         # deterministic


def test_no_patient_in_both_train_and_test(small: dict) -> None:
    """THE leakage guard: no patient's windows may appear in more than one split."""
    w = small["ds"]["windows"]
    per_patient = w.groupby("patient_id").split.nunique()
    assert (per_patient == 1).all()
    assert not set(w.patient_id[w.split == "train"]) & set(w.patient_id[w.split == "test"])
    assert_no_patient_leakage(w)


def test_leakage_check_catches_row_level_split(small: dict) -> None:
    w = small["ds"]["windows"].copy()
    w["split"] = np.where(np.arange(len(w)) % 5 == 0, "test", "train")   # a (wrong) row-level split
    with pytest.raises(PatientLeakageError):
        assert_no_patient_leakage(w)
