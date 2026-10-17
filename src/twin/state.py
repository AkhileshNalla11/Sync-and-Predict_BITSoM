"""The patient twin: static profile + recent sensor history + current hyper / hypo risk.

    ctx = TwinContext.load(cfg)          # loads data, model and alert thresholds once
    twin = ctx.twin("P0042")             # latest prediction time for that patient
    twin = ctx.twin("P0042", at="2026-01-10 21:00")
    ctx.risk_timeline("P0042")           # risk at every prediction time (for the dashboard)

The predictor is the LightGBM model named in config `twin.model` (CGM + wearables + EHR by default:
best hypo AUPRC in the ablation and well calibrated). Alert thresholds come from VALIDATION patients
so the test patients shown in the dashboard stay untouched. SYNTHETIC DATA, NOT FOR CLINICAL USE.
"""
from __future__ import annotations

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

TARGETS = ("hyper", "hypo")
SLEEP_NAMES = {0: "awake", 1: "light", 2: "deep", 3: "rem"}


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
    alert: dict[str, bool]               # risk >= alert threshold
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

        self.model_name = cfg["twin"]["model"]
        self.feature_set = self.model_name.removeprefix("lgbm_")
        models_dir = resolve_path(cfg["paths"]["models_dir"])
        self.boosters = {t: lgb.Booster(model_file=str(models_dir / f"{self.model_name}_{t}.txt")) for t in TARGETS}
        self.feature_names = list(FEATURE_SETS[self.feature_set])
        if self.boosters["hyper"].feature_name() != self.feature_names:
            raise ValueError("saved model features do not match FEATURE_SETS; re-run the baselines")
        self._window_feats = load_or_build_features(ds, cfg)
        self.thresholds = self._alert_thresholds(models_dir)

    @classmethod
    def load(cls, cfg: dict[str, Any] | None = None) -> "TwinContext":
        return cls(cfg or load_config())

    # ------------------------------------------------------------------ prediction
    def model_input(self, window_feats: pd.DataFrame, windows: pd.DataFrame,
                    static: pd.DataFrame | None = None) -> pd.DataFrame:
        """Assemble the model's input columns (static vector joined on patient_id)."""
        return feature_matrix(window_feats, windows, self.static if static is None else static, self.feature_set)

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {t: self.boosters[t].predict(X, num_iteration=self.boosters[t].best_iteration) for t in TARGETS}

    def _alert_thresholds(self, models_dir: Path) -> dict[str, float]:
        """Thresholds at the configured false-alert rate on VALIDATION onset windows (cached to JSON)."""
        far = self.cfg["twin"]["alert_false_alert_rate"]
        path = models_dir / f"{self.model_name}_alert_thresholds.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("false_alert_rate") == far and saved.get("n_windows") == len(self.windows):
                return {t: saved[t] for t in TARGETS}
        va = (self.windows.split == "val").to_numpy()
        probs = self.predict(self.model_input(self._window_feats[va], self.windows[va]))
        out = {}
        for t in TARGETS:
            onset = ~self.windows[va][f"in_event_{t}"].to_numpy(bool)
            out[t] = threshold_at_far(self.windows[va][t].to_numpy(int)[onset], probs[t][onset], far)
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
        """Predicted risks and alerts at every prediction time of one patient, plus observed outcomes."""
        rows = (self.windows.patient_id == patient_id).to_numpy()
        w = self.windows[rows]
        probs = self.predict(self.model_input(self._window_feats[rows], w))
        out = w[["timestamp", "anchor_idx", "hyper", "hypo", "in_event_hyper", "in_event_hypo", "nocturnal"]].copy()
        for t in TARGETS:
            out[f"{t}_risk"] = probs[t]
            out[f"{t}_alert"] = probs[t] >= self.thresholds[t]
        return out.reset_index(drop=True)

    def series_frame(self, patient_id: str) -> pd.DataFrame:
        """The patient's full sensor record as a readable frame."""
        s = self.series[self._pidx[patient_id]]
        df = pd.DataFrame(s, columns=list(SERIES_CHANNELS))
        df.insert(0, "timestamp", self.timestamps)
        df["sleep_stage"] = df.pop("sleep_code").map(SLEEP_NAMES)
        return df
