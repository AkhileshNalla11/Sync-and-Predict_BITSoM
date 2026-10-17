"""Stage 1 tests: synthetic EHR + wearable generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.ehr_features import generate_cohort
from src.data.synthetic_gen import generate_dataset, summarize

N_PATIENTS, N_DAYS = 40, 4


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def tables(cfg: dict) -> dict[str, pd.DataFrame]:
    return generate_dataset(cfg, N_PATIENTS, N_DAYS)


def test_shapes_and_schema(tables: dict[str, pd.DataFrame], cfg: dict) -> None:
    steps_per_day = 24 * 60 // cfg["sampling"]["interval_minutes"]
    w = tables["wearables"]
    assert len(w) == N_PATIENTS * N_DAYS * steps_per_day
    assert list(w.columns) == ["device_id", "timestamp", "cgm_mgdl", "steps", "heart_rate_bpm", "sleep_stage"]
    assert w.groupby("device_id").size().eq(N_DAYS * steps_per_day).all()
    assert w.timestamp.diff().dropna().isin([pd.Timedelta(minutes=5), pd.Timedelta(days=-N_DAYS) + pd.Timedelta(minutes=5)]).all()
    assert len(tables["ehr"]) == N_PATIENTS and tables["ehr"].patient_id.is_unique
    enr = tables["enrollment"]
    assert enr.patient_id.is_unique and enr.device_id.is_unique
    assert set(w.device_id) == set(enr.device_id)
    assert set(enr.patient_id) == set(tables["ehr"].patient_id)


def test_value_ranges(tables: dict[str, pd.DataFrame]) -> None:
    w = tables["wearables"]
    cgm = w.cgm_mgdl.dropna()
    assert cgm.between(40, 400).all()
    assert 0.005 < w.cgm_mgdl.isna().mean() < 0.15            # dropouts exist but are not dominant
    assert w.heart_rate_bpm.dropna().between(40, 190).all()
    assert (w.steps.dropna() >= 0).all()
    assert set(w.sleep_stage.dropna().unique()) <= {"awake", "light", "deep", "rem"}
    assert tables["ehr"].hba1c.between(5.7, 12.5).all()


def test_deterministic(cfg: dict, tables: dict[str, pd.DataFrame]) -> None:
    again = generate_dataset(cfg, N_PATIENTS, N_DAYS)
    for name in ("ehr", "enrollment", "wearables", "events"):
        pd.testing.assert_frame_equal(tables[name], again[name])


def test_metformin_only_rarely_hypo(tables: dict[str, pd.DataFrame], cfg: dict) -> None:
    s = summarize(tables, cfg)
    assert s["hypo_rate_other"] < 0.01
    assert s["hypo_rate_on_insulin_or_su"] > s["hypo_rate_other"]


def test_hba1c_drives_glucose(tables: dict[str, pd.DataFrame]) -> None:
    """By design, higher HbA1c -> higher simulated glucose (fusion signal exists)."""
    w = tables["wearables"].merge(tables["enrollment"][["device_id", "patient_id"]], on="device_id")
    mean_g = w.groupby("patient_id").cgm_mgdl.mean()
    a1c = tables["ehr"].set_index("patient_id").hba1c.loc[mean_g.index]
    assert np.corrcoef(a1c, mean_g)[0, 1] > 0.3


def test_meals_raise_glucose(tables: dict[str, pd.DataFrame]) -> None:
    """True glucose ~75 min after a large meal exceeds the pre-meal level on average."""
    truth = tables["sim_truth"].set_index(["patient_id", "timestamp"]).glucose_true_mgdl
    ev = tables["events"]
    meals = ev[(ev.event_type == "meal") & (ev.value > 60)]
    rises = []
    for _, m in meals.iterrows():
        try:
            rises.append(truth[(m.patient_id, m.timestamp + pd.Timedelta(minutes=75))] - truth[(m.patient_id, m.timestamp)])
        except KeyError:
            continue
    assert len(rises) > 50 and np.mean(rises) > 30


def test_ehr_cohort_plausible() -> None:
    ehr, latent = generate_cohort(2000, np.random.default_rng(0))
    assert 0.45 < (ehr.sex == "M").mean() < 0.65
    assert 0.05 < ehr.on_insulin.mean() < 0.30
    assert ehr.loc[ehr.egfr < 30, "on_metformin"].sum() == 0           # contraindicated
    assert not ehr.loc[~(ehr.on_insulin | ehr.on_sulfonylurea), "prior_severe_hypo_12m"].any()
    assert "drug_sensitivity" not in ehr.columns                        # latent stays hidden
    assert len(latent) == len(ehr)
