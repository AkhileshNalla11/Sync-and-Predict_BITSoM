"""Risk service: one FHIR Bundle in, the 2-hour risks out (as JSON and as FHIR RiskAssessment resources).

    svc = RiskService(model="lgbm_cgm_shanghai")      # or the synthetic-trained default (config twin.model)
    result = svc.assess(bundle)                         # dict: risks, thresholds, top drivers, data checks
    fhir = svc.to_fhir(result)                          # Bundle of RiskAssessment resources

The window is built exactly as in training: the last 6 h before `now` (default: the latest CGM reading) on a
5-min grid. Readings are snapped to the nearest grid point; sparser sensors (e.g. 15-min CGM) are linearly
interpolated between readings at most 30 min apart, as for ShanghaiT2DM, on which the real-patient-trained
model was fitted. A window with less CGM than the training minimum (config windows.min_history_coverage) is
refused rather than guessed.

The response says whether each risk is at or above the model's alert threshold. The episode alert policy
(one alert per episode, refractory period) needs the alert history over time, so it is the caller's to apply
across successive calls (src/twin/alerts.py). Not a medical device, not for clinical use.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import resolve_path, runtime_config
from src.data.ehr_features import STATIC_FEATURES, StaticEncoder, encode_static
from src.data.shanghai import MAX_INTERP_MIN
from src.data.wearable_features import SERIES_CHANNELS, handcrafted_features, window_sizes
from src.interop.fhir import SNOMED, PatientRecord, parse_bundle
from src.models.baselines import FEATURE_SETS, feature_matrix
from src.twin.explain import explain
from src.twin.state import EXTERNAL_MODELS

TARGETS = ("hyper", "hypo")
CHANNELS = ("cgm_mgdl", "steps", "heart_rate_bpm", "sleep_stage")          # order of SERIES_CHANNELS
OUTCOME = {"hypo": ("302866003", "Hypoglycaemia", "Any CGM reading < 70 mg/dL in the next 2 hours"),
           "hyper": ("80394007", "Hyperglycaemia", "Any CGM reading > 180 mg/dL in the next 2 hours")}
DISCLAIMER = ("Research prototype. Not a medical device and not for clinical use. Risk scores have not been "
              "clinically validated; continuous glucose monitor alarms must stay on.")


class InsufficientData(ValueError):
    """The bundle does not hold enough CGM to score the window as the model was trained."""


class RiskService:
    def __init__(self, cfg: dict[str, Any] | None = None, model: str | None = None):
        self.cfg = cfg or runtime_config()
        self.model_name = model or self.cfg["twin"]["model"]
        self.feature_set = EXTERNAL_MODELS.get(self.model_name, self.model_name.removeprefix("lgbm_"))
        models_dir = resolve_path(self.cfg["paths"]["models_dir"])
        self.boosters = {t: lgb.Booster(model_file=str(models_dir / f"{self.model_name}_{t}.txt")) for t in TARGETS}
        self.feature_names = list(FEATURE_SETS[self.feature_set])
        saved = json.loads((models_dir / f"{self.model_name}_alert_thresholds.json").read_text(encoding="utf-8"))
        self.thresholds = {t: float(saved[t]) for t in TARGETS}
        self.trained_on = saved.get("fitted_on", "synthetic patients (this project's simulator)")
        self.uses_static = any(f in STATIC_FEATURES for f in self.feature_names)
        self.encoder = (StaticEncoder.from_dict(json.loads((resolve_path(self.cfg["paths"]["processed_dir"]) /
                                                            "static_encoder.json").read_text(encoding="utf-8")))
                        if self.uses_static else None)
        self.dt = self.cfg["sampling"]["interval_minutes"]
        self.history = window_sizes(self.cfg)[0]

    # ------------------------------------------------------------------ window
    def window(self, readings: pd.DataFrame, now: pd.Timestamp) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, float]]:
        """(1, history, 4) raw window ending at `now`, its grid, and data checks."""
        grid = pd.date_range(end=now, periods=self.history, freq=f"{self.dt}min")
        raw = np.full((1, self.history, len(SERIES_CHANNELS)), np.nan, np.float32)
        info: dict[str, float] = {}
        for c, ch in enumerate(CHANNELS):
            r = readings[readings.channel == ch]
            slot = np.round((r.timestamp - grid[0]).dt.total_seconds().to_numpy() / 60 / self.dt).astype(int) if len(r) else np.zeros(0, int)
            ok = (slot >= 0) & (slot < self.history) & np.isfinite(r.value.to_numpy(float))
            col = pd.Series(np.nan, index=range(self.history))
            for s, v in zip(slot[ok], r.value.to_numpy(float)[ok]):      # later readings win within a slot
                col[s] = v
            if ch == "cgm_mgdl":
                t_obs = np.sort(r.timestamp[ok].to_numpy())
                native = float(np.median(np.diff(t_obs).astype("timedelta64[s]").astype(float)) / 60) if len(t_obs) > 1 else np.nan
                info.update(cgm_readings=int(ok.sum()), native_interval_min=native)
                if np.isfinite(native) and native > self.dt:            # sparse sensor: bridge short gaps only
                    known = col.dropna().index.to_numpy()
                    filled = col.interpolate(limit_area="inside")
                    if len(known) > 1:
                        nxt = np.searchsorted(known, filled.index.to_numpy())
                        prev = np.clip(nxt - 1, 0, len(known) - 1)
                        nxt = np.clip(nxt, 0, len(known) - 1)
                        gap_min = (known[nxt] - known[prev]) * self.dt
                        filled[(gap_min > MAX_INTERP_MIN) & col.isna().to_numpy()] = np.nan
                    col = filled
                info["cgm_coverage"] = float(col.notna().mean())
            raw[0, :, c] = col.to_numpy(np.float32)
        return raw, grid, info

    # ------------------------------------------------------------------ assessment
    def assess(self, bundle: dict[str, Any] | PatientRecord, now: str | pd.Timestamp | None = None) -> dict[str, Any]:
        rec = bundle if isinstance(bundle, PatientRecord) else parse_bundle(bundle, None if now is None else pd.Timestamp(now))
        cgm_t = rec.readings.timestamp[rec.readings.channel == "cgm_mgdl"]
        if not len(cgm_t):
            raise InsufficientData("the bundle has no CGM Observations (LOINC 99504-3)")
        now_ts = pd.Timestamp(now).tz_localize(None) if now is not None and pd.Timestamp(now).tzinfo else (
            pd.Timestamp(now) if now is not None else cgm_t.max())
        raw, grid, info = self.window(rec.readings, now_ts)
        need = float(self.cfg["windows"]["min_history_coverage"])
        if info["cgm_coverage"] < need:
            raise InsufficientData(f"only {info['cgm_coverage']:.0%} of the last {self.history * self.dt // 60} h has CGM; "
                                   f"the model needs at least {need:.0%}")
        feats = handcrafted_features(raw, np.array([self.history - 1]), grid[0].hour + grid[0].minute / 60, self.dt, self.cfg)
        if self.uses_static:
            static_raw = encode_static(pd.DataFrame([{**rec.profile, "patient_id": rec.patient_id}]))
            X = feature_matrix(feats, pd.DataFrame({"patient_id": [rec.patient_id]}), self.encoder.transform(static_raw),
                               self.feature_set)
            static_row = static_raw.iloc[0]
            missing = [f for f in STATIC_FEATURES if not np.isfinite(float(static_row[f]))]
        else:
            X, static_row, missing = feats[self.feature_names], pd.Series(dtype=float), []
        snapshot = SimpleNamespace(features=X, static_raw=static_row)
        risks = {}
        for t in TARGETS:
            b = self.boosters[t]
            p = float(b.predict(X, num_iteration=b.best_iteration)[0])
            d = explain(self, snapshot, t, top_k=3)
            risks[t] = {"probability": p, "alert_threshold": self.thresholds[t], "above_alert_threshold": p >= self.thresholds[t],
                        "drivers": [{"factor": r.label, "value": r.value, "effect": r.direction, "log_odds": round(r.contribution, 4)}
                                    for r in d.itertuples()]}
        warnings = list(rec.warnings)
        if self.uses_static and missing:
            warnings.append(f"EHR fields missing from the bundle (scored as unknown): {', '.join(missing)}")
        return {"patient_id": rec.patient_id, "assessed_at": now_ts.isoformat(), "horizon_hours": self.cfg["windows"]["horizon_hours"],
                "model": {"name": self.model_name, "inputs": "CGM + time of day" if not self.uses_static else
                          "CGM + wearables + EHR", "trained_on": self.trained_on},
                "risks": risks, "data": {**info, "history_hours": self.history * self.dt / 60}, "warnings": warnings,
                "disclaimer": DISCLAIMER}

    # ------------------------------------------------------------------ FHIR output
    def to_fhir(self, result: dict[str, Any]) -> dict[str, Any]:
        """A collection Bundle with one RiskAssessment per event, ready to post back to a FHIR server."""
        start = pd.Timestamp(result["assessed_at"])
        end = start + pd.Timedelta(hours=result["horizon_hours"])
        entries = []
        for t in TARGETS:
            r = result["risks"][t]
            code, name, rule = OUTCOME[t]
            why = "; ".join(f"{d['factor']} {d['value']} ({d['effect']})" for d in r["drivers"])
            ra = {
                "resourceType": "RiskAssessment", "id": f"{result['patient_id']}-{t}-{start:%Y%m%d%H%M}", "status": "final",
                "subject": {"reference": f"Patient/{result['patient_id']}"}, "occurrenceDateTime": start.isoformat(),
                "method": {"text": f"{result['model']['name']}: LightGBM on {result['model']['inputs']}, "
                                   f"trained on {result['model']['trained_on']}"},
                "prediction": [{
                    "outcome": {"coding": [{"system": SNOMED, "code": code, "display": name}], "text": rule},
                    "probabilityDecimal": round(r["probability"], 4),
                    "qualitativeRisk": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                                                    "code": "high" if r["above_alert_threshold"] else "low"}]},
                    "whenPeriod": {"start": start.isoformat(), "end": end.isoformat()},
                    "rationale": f"Alert threshold {r['alert_threshold']:.3f}. Top drivers: {why}.",
                }],
                "note": [{"text": DISCLAIMER}],
            }
            entries.append({"fullUrl": f"urn:uuid:riskassessment-{ra['id']}", "resource": ra})
        return {"resourceType": "Bundle", "type": "collection", "timestamp": start.isoformat(), "entry": entries}
