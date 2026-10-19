"""The patient twin: static profile + recent sensor history + current hyper / hypo risk.

    ctx = TwinContext.load(cfg)          # loads data, model and alert thresholds once
    twin = ctx.twin("P0042")             # latest prediction time for that patient
    twin = ctx.twin("P0042", at="2026-01-10 21:00")
    ctx.risk_timeline("P0042")           # risk and policy alerts at every prediction time (for the dashboard)

The predictor is the LightGBM model named in config `twin.model` (CGM + wearables + EHR by default:
best hypo AUPRC in the ablation and well calibrated). Alert thresholds come from VALIDATION patients
so the test patients shown in the dashboard stay untouched. `ctx.with_model("lgbm_cgm_shanghai")` returns
a context that shares the data but predicts with the CGM-only model trained on real patients
(src/eval/real_model.py); its thresholds were set on real patients and are read from its JSON file. Over time, alerts follow the alert policy
in config `twin.alert_policy` (src/twin/alerts.py): one alert per risk episode, not one per window.
SYNTHETIC DATA, NOT FOR CLINICAL USE.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.build_dataset import load_processed
from src.data.ehr_features import StaticEncoder, encode_static
from src.data.wearable_features import SERIES_CHANNELS, extract_windows, handcrafted_features, window_sizes
from src.eval.metrics import threshold_at_far
from src.models.baselines import FEATURE_SETS, feature_matrix, load_or_build_features
from src.twin.alerts import AlertPolicy, policy_alerts, policy_far, policy_from_spec

TARGETS = ("hyper", "hypo")
SLEEP_NAMES = {0: "awake", 1: "light", 2: "deep", 3: "rem"}
# Models trained outside the synthetic pipeline: name -> feature set. Their thresholds are fixed at training time.
EXTERNAL_MODELS = {"lgbm_cgm_shanghai": "cgm"}


@dataclass
class PatientTwin:
    """Snapshot of one virtual patient at one prediction time."""

    patient_id: str
    anchor_idx: int
    time: pd.Timestamp
    profile: dict[str, Any]              # raw EHR record (for display)
    static_raw: pd.Series                # encoded but unscaled static features (for display / explanations)
    history: pd.DataFrame                # last 6 h of sensor data
    features: pd.DataFrame               # one-row model input (handcrafted window features + scaled static)
    risk: dict[str, float]               # predicted probabilities for the next 2 h
    alert: dict[str, bool]               # risk >= alert threshold (snapshot; the timeline applies the alert policy)
    outcome: dict[str, bool] | None = field(default=None)   # what actually happened (synthetic ground truth)


class TwinContext:
    """Everything needed to build twins, loaded once (cache this object in the dashboard)."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        ds = load_processed(cfg)
        self.series: np.ndarray = ds["series"]
        self.timestamps: pd.DatetimeIndex = ds["timestamps"]
        self.patient_ids = [str(p) for p in ds["patient_ids"]]
        self._pidx = {p: i for i, p in enumerate(self.patient_ids)}
        self.windows: pd.DataFrame = ds["windows"]
        self.static: pd.DataFrame = ds["static"]
        self.encoder: StaticEncoder = ds["encoder"]
        self.splits: pd.DataFrame = ds["splits"]
        raw = resolve_path(cfg["paths"]["raw_dir"])
        self.ehr = pd.read_csv(raw / "ehr.csv", keep_default_na=False).set_index("patient_id")
        self.history_len = window_sizes(cfg)[0]
        self.dt = cfg["sampling"]["interval_minutes"]
        ts0 = self.timestamps[0]
        self.start_hour = ts0.hour + ts0.minute / 60

        self._window_feats = load_or_build_features(ds, cfg)
        self._use_model(cfg["twin"]["model"])

    def _use_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.feature_set = EXTERNAL_MODELS.get(model_name, model_name.removeprefix("lgbm_"))
        models_dir = resolve_path(self.cfg["paths"]["models_dir"])
        self.boosters = {t: lgb.Booster(model_file=str(models_dir / f"{model_name}_{t}.txt")) for t in TARGETS}
        self.feature_names = list(FEATURE_SETS[self.feature_set])
        if self.boosters["hyper"].feature_name() != self.feature_names:
            raise ValueError("saved model features do not match FEATURE_SETS; re-run the baselines")
        self.thresholds = self._alert_thresholds(models_dir)
        self.policies: dict[str, AlertPolicy] = {t: policy_from_spec(self.cfg["twin"]["alert_policy"][t], self.thresholds[t])
                                                 for t in TARGETS}

    @classmethod
    def load(cls, cfg: dict[str, Any] | None = None) -> "TwinContext":
        return cls(cfg or load_config())

    def with_model(self, model_name: str) -> "TwinContext":
        """A context sharing this one's data (no copy) but predicting with another saved model."""
        if model_name == self.model_name:
            return self
        other = copy.copy(self)
        other._use_model(model_name)
        return other

    @staticmethod
    def model_available(cfg: dict[str, Any], model_name: str) -> bool:
        models_dir = resolve_path(cfg["paths"]["models_dir"])
        files = [models_dir / f"{model_name}_{t}.txt" for t in TARGETS]
        if model_name in EXTERNAL_MODELS:
            files.append(models_dir / f"{model_name}_alert_thresholds.json")
        return all(f.exists() for f in files)

    # ------------------------------------------------------------------ prediction
    def model_input(self, window_feats: pd.DataFrame, windows: pd.DataFrame,
                    static: pd.DataFrame | None = None) -> pd.DataFrame:
        """Assemble the model's input columns (static vector joined on patient_id)."""
        return feature_matrix(window_feats, windows, self.static if static is None else static, self.feature_set)

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {t: self.boosters[t].predict(X, num_iteration=self.boosters[t].best_iteration) for t in TARGETS}

    def split_scores(self, split: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """All prediction windows of one patient split and the model's risk for each."""
        rows = (self.windows.split == split).to_numpy()
        w = self.windows[rows]
        return w, self.predict(self.model_input(self._window_feats[rows], w))

    def _alert_thresholds(self, models_dir: Path) -> dict[str, float]:
        """Per-target thresholds at the policy's false-alert rate on VALIDATION onset windows (cached to JSON)."""
        far = policy_far(self.cfg)
        path = models_dir / f"{self.model_name}_alert_thresholds.json"
        if self.model_name in EXTERNAL_MODELS:                 # set on real patients when the model was trained
            if not path.exists():
                raise FileNotFoundError(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            return {t: float(saved[t]) for t in TARGETS}
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("false_alert_rate") == far and saved.get("n_windows") == len(self.windows):
                return {t: saved[t] for t in TARGETS}
        w, probs = self.split_scores("val")
        out = {}
        for t in TARGETS:
            onset = ~w[f"in_event_{t}"].to_numpy(bool)
            out[t] = threshold_at_far(w[t].to_numpy(int)[onset], probs[t][onset], far[t])
        path.write_text(json.dumps({**out, "false_alert_rate": far, "n_windows": len(self.windows)}, indent=2), encoding="utf-8")
        return out

    # ------------------------------------------------------------------ twins
    def patients(self, split: str | None = "test") -> list[str]:
        s = self.splits if split is None else self.splits[self.splits.split == split]
        return sorted(s.patient_id)

    def patient_windows(self, patient_id: str) -> pd.DataFrame:
        return self.windows[self.windows.patient_id == patient_id]

    def history_frame(self, series_row: np.ndarray, anchor_idx: int) -> pd.DataFrame:
        """Readable 6 h history ending at the anchor, from one patient's (T, C) series."""
        sl = slice(anchor_idx - self.history_len + 1, anchor_idx + 1)
        h = pd.DataFrame(series_row[sl], columns=list(SERIES_CHANNELS))
        h.insert(0, "timestamp", self.timestamps[sl])
        h["sleep_stage"] = h.pop("sleep_code").map(SLEEP_NAMES)
        return h

    def build_twin(self, patient_id: str, anchor_idx: int, series_row: np.ndarray | None = None,
                   profile: dict[str, Any] | None = None) -> PatientTwin:
        """Twin at a given anchor. `series_row` / `profile` override the stored data (used by what-if)."""
        p = self._pidx[patient_id]
        observed = series_row is None and profile is None      # outcome is only known for the real record
        series_row = self.series[p] if series_row is None else series_row
        profile = self.ehr.loc[patient_id].to_dict() if profile is None else profile
        raw = extract_windows(series_row[None], np.array([0]), np.array([anchor_idx]), self.history_len)
        feats = handcrafted_features(raw, np.array([anchor_idx]), self.start_hour, self.dt, self.cfg)
        static_raw = encode_static(pd.DataFrame([{**profile, "patient_id": patient_id}]))
        static = self.encoder.transform(static_raw)
        X = self.model_input(feats, pd.DataFrame({"patient_id": [patient_id]}), static)
        risk = {t: float(v[0]) for t, v in self.predict(X).items()}
        w = self.patient_windows(patient_id)
        row = w[w.anchor_idx == anchor_idx]
        outcome = {t: bool(row[t].iloc[0]) for t in TARGETS} if observed and len(row) else None
        return PatientTwin(patient_id=patient_id, anchor_idx=int(anchor_idx), time=self.timestamps[anchor_idx],
                           profile=profile, static_raw=static_raw.iloc[0], history=self.history_frame(series_row, anchor_idx),
                           features=X, risk=risk, alert={t: risk[t] >= self.thresholds[t] for t in TARGETS},
                           outcome=outcome)

    def twin(self, patient_id: str, at: str | pd.Timestamp | None = None) -> PatientTwin:
        """Twin at the latest prediction time <= `at` (default: the patient's last prediction time)."""
        w = self.patient_windows(patient_id)
        if at is None:
            anchor = int(w.anchor_idx.max())
        else:
            idx = int(self.timestamps.searchsorted(pd.Timestamp(at), side="right")) - 1
            valid = w.anchor_idx[w.anchor_idx <= idx]
            anchor = int(valid.max() if len(valid) else w.anchor_idx.min())
        return self.build_twin(patient_id, anchor)

    def risk_timeline(self, patient_id: str) -> pd.DataFrame:
        """Predicted risks and alerts at every prediction time of one patient, plus observed outcomes.

        `<t>_above`: risk at or above the alert threshold. `<t>_alert`: an alert fired here under the
        alert policy (once per risk episode; repeats suppressed by refractory period and hysteresis).
        """
        rows = (self.windows.patient_id == patient_id).to_numpy()
        w = self.windows[rows]
        probs = self.predict(self.model_input(self._window_feats[rows], w))
        out = w[["timestamp", "anchor_idx", "hyper", "hypo", "in_event_hyper", "in_event_hypo", "nocturnal"]].copy()
        for t in TARGETS:
            out[f"{t}_risk"] = probs[t]
            out[f"{t}_above"] = probs[t] >= self.thresholds[t]
            out[f"{t}_alert"] = policy_alerts(w, probs[t], self.policies[t], t)
        return out.reset_index(drop=True)

    def series_frame(self, patient_id: str) -> pd.DataFrame:
        """The patient's full sensor record as a readable frame."""
        s = self.series[self._pidx[patient_id]]
        df = pd.DataFrame(s, columns=list(SERIES_CHANNELS))
        df.insert(0, "timestamp", self.timestamps)
        df["sleep_stage"] = df.pop("sleep_code").map(SLEEP_NAMES)
        return df
