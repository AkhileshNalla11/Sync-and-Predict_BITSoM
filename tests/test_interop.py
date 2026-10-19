"""FHIR in and out, the risk service and the API. Runs on the committed demo bundle, so it runs in CI too."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.interop.fhir import LOINC, parse_bundle, to_bundle

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "fhir" / "P0014_bundle.json"
BUNDLE_OK = (ROOT / "demo_bundle" / "processed" / "series.npz").exists()
needs_bundle = pytest.mark.skipif(not BUNDLE_OK, reason="demo bundle not built")
NOW = pd.Timestamp("2026-01-18 09:55")
EHR = {"age": 66, "sex": "F", "weight_kg": 66.1, "bmi": 28.2, "diabetes_duration_years": 6.2, "hba1c": 7.1, "egfr": 100.0,
       "ldl_mgdl": 76.0, "triglycerides_mgdl": 178.0, "hypertension": True, "dyslipidemia": False,
       "coronary_artery_disease": False, "family_history_diabetes": True, "on_metformin": True, "on_sulfonylurea": True,
       "on_insulin": True, "insulin_regimen": "premix_bid", "insulin_total_daily_units": 24.0, "sulfonylurea_dose_mg": 3.0,
       "sulfonylurea_frequency": "BD", "metformin_dose_mg": 1000, "prior_severe_hypo_12m": True}


def _history(n: int = 72) -> pd.DataFrame:
    t = pd.date_range(end=NOW, periods=n, freq="5min")
    g = 150 - np.arange(n) * 1.0
    return pd.DataFrame({"timestamp": t, "cgm_mgdl": g, "steps": 10.0, "heart_rate_bpm": 72.0, "sleep_stage": "awake"})


def _entries(b: dict, kind: str) -> list[dict]:
    return [e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == kind]


@pytest.fixture()
def bundle_cfg(monkeypatch):
    monkeypatch.setenv("TWIN_USE_DEMO_BUNDLE", "1")
    from src.config import runtime_config
    return runtime_config()


# --------------------------------------------------------------------------- parsing
def test_profile_survives_a_round_trip() -> None:
    rec = parse_bundle(to_bundle("X1", EHR, _history(), NOW))
    p = rec.profile
    assert rec.patient_id == "X1" and not rec.warnings
    assert p["age"] == 66 and p["sex"] == "F"
    assert p["diabetes_duration_years"] == pytest.approx(6.2, abs=1e-6)
    for k in ("hba1c", "egfr", "ldl_mgdl", "triglycerides_mgdl", "bmi", "weight_kg"):
        assert p[k] == pytest.approx(EHR[k])
    assert (p["metformin_dose_mg"], p["sulfonylurea_dose_mg"], p["sulfonylurea_frequency"]) == (1000, 3.0, "BD")
    assert (p["insulin_total_daily_units"], p["insulin_regimen"]) == (24.0, "premix_bid")
    assert p["hypertension"] and p["family_history_diabetes"] and p["prior_severe_hypo_12m"] and not p["dyslipidemia"]
    counts = rec.readings.channel.value_counts()
    assert counts["cgm_mgdl"] == counts["steps"] == counts["heart_rate_bpm"] == counts["sleep_stage"] == 72


def test_units_codes_and_statuses() -> None:
    b = to_bundle("X2", {**EHR, "on_insulin": False, "insulin_regimen": "", "prior_severe_hypo_12m": False}, _history(), NOW)
    for o in _entries(b, "Observation"):
        code = o["code"]["coding"][0]["code"]
        if code == "99504-3":                                                   # CGM sent in mmol/L
            o["valueQuantity"] = {"value": o["valueQuantity"]["value"] / 18.016, "unit": "mmol/L"}
        if code == "4548-4":                                                    # HbA1c sent in mmol/mol
            o["valueQuantity"] = {"value": 54.0, "unit": "mmol/mol"}
        if code == "13457-7":                                                   # LDL sent in mmol/L
            o["valueQuantity"] = {"value": 2.0, "unit": "mmol/L"}
    for m in _entries(b, "MedicationStatement"):
        if m["medicationCodeableConcept"]["text"] == "Glimepiride":
            m["medicationCodeableConcept"]["text"] = "Gliclazide MR"            # no dose conversion: on/off only
    _entries(b, "Condition")[1]["clinicalStatus"]["coding"][0]["code"] = "resolved"   # hypertension resolved
    rec = parse_bundle(b)
    cgm = rec.readings[rec.readings.channel == "cgm_mgdl"].value.to_numpy()
    assert cgm[-1] == pytest.approx(150 - 71, rel=1e-9)
    assert rec.profile["hba1c"] == pytest.approx(54 / 10.929 + 2.15)
    assert rec.profile["ldl_mgdl"] == pytest.approx(2.0 * 38.67)
    assert rec.profile["on_sulfonylurea"] and np.isnan(rec.profile["sulfonylurea_dose_mg"])
    assert any("gliclazide" in w for w in rec.warnings)
    assert not rec.profile["hypertension"] and not rec.profile["on_insulin"]


def test_rejects_bundles_without_one_patient() -> None:
    with pytest.raises(ValueError):
        parse_bundle({"resourceType": "Bundle", "entry": []})
    with pytest.raises(ValueError):
        parse_bundle({"resourceType": "Patient"})


# --------------------------------------------------------------------------- service
@needs_bundle
@pytest.mark.parametrize("model", [None, "lgbm_cgm_shanghai"])
def test_fhir_path_reproduces_the_dashboard_risk(bundle_cfg, model) -> None:
    from src.interop.service import RiskService
    from src.twin.state import TwinContext
    ctx = TwinContext.load(bundle_cfg)
    ctx = ctx.with_model(model or bundle_cfg["twin"]["model"])
    svc = RiskService(bundle_cfg, model)
    pid = ctx.patients("test")[0]
    anchors = ctx.patient_windows(pid).anchor_idx.to_numpy()
    for anchor in anchors[[3, len(anchors) // 2, -1]]:
        twin = ctx.build_twin(pid, int(anchor))
        res = svc.assess(to_bundle(pid, ctx.ehr.loc[pid].to_dict(), twin.history, twin.time))
        for t in ("hyper", "hypo"):
            assert res["risks"][t]["probability"] == pytest.approx(twin.risk[t], abs=1e-9)
            assert res["risks"][t]["above_alert_threshold"] == (twin.risk[t] >= ctx.thresholds[t])
            assert len(res["risks"][t]["drivers"]) == 3


@needs_bundle
def test_sparse_sensor_and_too_little_data(bundle_cfg) -> None:
    from src.interop.service import InsufficientData, RiskService
    svc = RiskService(bundle_cfg, "lgbm_cgm_shanghai")
    b = to_bundle("X3", EHR, _history(), NOW)
    sparse = copy.deepcopy(b)
    keep = {f"cgm-{i}" for i in range(71, -1, -3)}                              # a 15-min sensor
    sparse["entry"] = [e for e in sparse["entry"] if not e["resource"]["id"].startswith("cgm-") or e["resource"]["id"] in keep]
    res = svc.assess(sparse)
    assert res["data"]["native_interval_min"] == 15 and res["data"]["cgm_coverage"] > 0.95   # bridged to 5 min
    short = copy.deepcopy(b)
    short["entry"] = [e for e in short["entry"] if not e["resource"]["id"].startswith("cgm-")
                      or int(e["resource"]["id"].split("-")[1]) >= 60]           # only the last hour
    with pytest.raises(InsufficientData):
        svc.assess(short)


# --------------------------------------------------------------------------- API
@needs_bundle
def test_api_endpoints(bundle_cfg) -> None:
    from fastapi.testclient import TestClient
    from src.api.app import app, service
    service.cache_clear()
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    models = c.get("/v1/models").json()
    assert models["real"]["inputs"] == "CGM + time of day" and models["real"]["trained_on"] == "ShanghaiT2DM"
    body = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    r = c.post("/v1/risk", json=body)
    assert r.status_code == 200 and set(r.json()["risks"]) == {"hyper", "hypo"}
    assert c.post("/v1/risk?model=synthetic", json=body).json()["model"]["inputs"] == "CGM + wearables + EHR"
    fhir = c.post("/v1/risk/fhir", json=body).json()
    ras = [e["resource"] for e in fhir["entry"]]
    assert [ra["resourceType"] for ra in ras] == ["RiskAssessment", "RiskAssessment"]
    assert {ra["prediction"][0]["outcome"]["coding"][0]["code"] for ra in ras} == {"302866003", "80394007"}
    assert 0 <= ras[1]["prediction"][0]["probabilityDecimal"] <= 1
    assert c.post("/v1/risk", json={"resourceType": "Bundle", "entry": []}).status_code == 400
    no_cgm = {**body, "entry": [e for e in body["entry"] if not e["resource"]["id"].startswith("cgm-")]}
    assert c.post("/v1/risk", json=no_cgm).status_code == 422
    service.cache_clear()


def test_example_bundle_is_synthetic_and_coded() -> None:
    b = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    pat = _entries(b, "Patient")[0]
    assert pat["meta"]["tag"][0]["code"] == "SYNTHETIC"
    codes = {o["code"]["coding"][0]["code"] for o in _entries(b, "Observation") if o["code"]["coding"][0]["system"] == LOINC}
    assert {"99504-3", "4548-4", "62238-1", "55423-8", "8867-4"} <= codes


# --------------------------------------------------------------------------- simulator realism metrics
def test_realism_metrics_on_known_series() -> None:
    from src.eval.simulator_realism import patient_metrics
    t = pd.date_range("2026-01-01", periods=96 * 2, freq="15min")
    zigzag = pd.Series(np.where(np.arange(len(t)) % 2 == 0, 120.0, 130.0), index=t)
    m, traj, _ = patient_metrics(zigzag)
    assert m["reversal_rate"] == 100 and m["tir"] == 100 and m["hypo_per_day"] == 0 and not traj
    fall = np.full(len(t), 130.0)
    fall[100:112] = 130 - np.arange(12) * 7.5                                   # 7.5 mg/dL per 15 min = 0.5 / min
    fall[112:116] = 47.5                                                        # through 70 at a steady rate, then flat
    m, traj, _ = patient_metrics(pd.Series(fall, index=t))
    assert m["hypo_per_day"] == pytest.approx(0.5) and len(traj) == 1
    assert m["pre_hypo_slope"] == pytest.approx(-0.5)
