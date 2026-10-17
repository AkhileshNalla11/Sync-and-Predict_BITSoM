"""Loader for CGMacros v1.0.0 (PhysioNet), used only for the external sanity check.

    https://physionet.org/content/cgmacros/  (CC BY-NC-SA 4.0; never committed to this repo)

Expected layout after extracting the CSVs from CGMacros_dateshifted365.zip (photos are not needed):

    data/external/cgmacros/CGMacros/bio.csv                       one row per `subject`, lab HbA1c
    data/external/cgmacros/CGMacros/CGMacros-0XX/CGMacros-0XX.csv 1-min rows: Timestamp, Libre GL, Dexcom GL, ...

Choices (documented in docs/external_validation.md):
  * One sensor for everyone: Dexcom G6 Pro by default. It samples every 5 min like our synthetic CGM;
    the Libre Pro samples every 15 min. Both are already linearly interpolated to 1 min in the files.
  * Resampling: the value at each 5-min clock mark, on a regular grid from the first to the last reading.
    Missing rows become NaN (unobserved), as in our synthetic series.
  * Bridged gaps: the published 1-min values are interpolated, so a dropout shows up as an exactly
    straight line. Straight stretches of >= `bridged_gap_min` minutes are set to NaN. This may also remove
    a few genuinely flat periods; the gap report counts how much was masked.
  * Glycaemic status comes from lab HbA1c with the dataset's own cut-offs:
    healthy < 5.7 %, prediabetes 5.7-6.4 %, T2D > 6.4 %.

The files are untrusted input: only the needed columns are read, values are parsed as numbers, and
readings outside the sensors' 40-400 mg/dL range are dropped and counted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.wearable_features import SERIES_CHANNELS, build_feature_table, label_windows

SENSOR_COLUMNS = {"dexcom": "Dexcom GL", "libre": "Libre GL"}
DEFAULT_SENSOR = "dexcom"
SENSOR_RANGE_MGDL = (40.0, 400.0)
HBA1C_COLUMN = "A1c PDL (Lab)"
STATUSES = ("t2d", "prediabetes", "healthy")
BRIDGED_GAP_MIN = 30


def glycaemic_status(hba1c: float) -> str:
    """Dataset cut-offs on lab HbA1c (%)."""
    if not np.isfinite(hba1c):
        return "unknown"
    return "healthy" if hba1c < 5.7 else ("prediabetes" if hba1c <= 6.4 else "t2d")


def participant_id(subject: int) -> str:
    return f"CGMacros-{int(subject):03d}"


def find_root(data_dir: str | Path) -> Path:
    """Accept either the folder that holds bio.csv or its parent (the zip extracts into CGMacros/)."""
    d = Path(data_dir)
    for cand in (d, d / "CGMacros"):
        if (cand / "bio.csv").is_file():
            return cand
    raise FileNotFoundError(f"bio.csv not found under {d} (see docs/external_validation.md for the download)")


def load_bio(root: str | Path) -> pd.DataFrame:
    """Participants indexed by id (CGMacros-0XX) with age, gender, BMI, lab HbA1c and status."""
    bio = pd.read_csv(Path(root) / "bio.csv")
    bio.columns = bio.columns.str.strip()
    out = pd.DataFrame({
        "participant_id": [participant_id(s) for s in pd.to_numeric(bio["subject"], errors="raise")],
        "age": pd.to_numeric(bio.get("Age"), errors="coerce"),
        "gender": bio.get("Gender"),
        "bmi": pd.to_numeric(bio.get("BMI"), errors="coerce"),
        "hba1c": pd.to_numeric(bio[HBA1C_COLUMN], errors="coerce"),
    })
    out["status"] = out.hba1c.map(glycaemic_status)
    return out.set_index("participant_id")


def read_sensor(path: str | Path, sensor: str = DEFAULT_SENSOR) -> tuple[pd.Series, int]:
    """Raw 1-min readings of one sensor (NaN rows dropped) and the number of out-of-range values removed."""
    col = SENSOR_COLUMNS[sensor]
    df = pd.read_csv(path, usecols=lambda c: c.strip() in ("Timestamp", col))
    df.columns = df.columns.str.strip()
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    v = pd.to_numeric(df[col], errors="coerce")
    s = pd.Series(v.to_numpy(float), index=ts).loc[lambda x: x.index.notna()].dropna()
    s = s[~s.index.duplicated(keep="first")].sort_index()
    lo, hi = SENSOR_RANGE_MGDL
    bad = (s < lo) | (s > hi)
    return s[~bad], int(bad.sum())


def mask_bridged_gaps(s1: pd.Series, min_minutes: int = BRIDGED_GAP_MIN) -> tuple[pd.Series, int]:
    """Set exactly linear stretches of >= `min_minutes` on the 1-min grid to NaN; returns minutes masked."""
    s = s1.asfreq("1min")
    v = s.to_numpy(float)
    straight = np.zeros(len(v), bool)
    if len(v) >= 3:
        d2 = np.abs(np.diff(v, 2))
        straight[1:-1] = d2 < 1e-6                     # NaN comparisons are False, so gaps break runs
    # interior points of a straight run of length L span L + 1 minutes between its two end readings
    mask = np.zeros(len(v), bool)
    start = None
    for i, x in enumerate(np.append(straight, False)):
        if x and start is None:
            start = i
        elif not x and start is not None:
            if i - start + 1 >= min_minutes:
                mask[start:i] = True
            start = None
    v = v.copy()
    v[mask] = np.nan
    return pd.Series(v, index=s.index), int(mask.sum())


def resample_5min(s1: pd.Series, dt_min: int = 5) -> pd.Series:
    """Values at each `dt_min` clock mark on a regular grid spanning the observed readings."""
    s1 = s1.dropna()
    if s1.empty:
        return pd.Series(dtype=float)
    grid = pd.date_range(s1.index[0].ceil(f"{dt_min}min"), s1.index[-1].floor(f"{dt_min}min"), freq=f"{dt_min}min")
    return s1.reindex(grid)


def gap_report(s5: pd.Series, dt_min: int = 5) -> dict[str, float]:
    """Coverage diagnostics of a regular 5-min series (NaN = unobserved)."""
    obs = s5.notna().to_numpy()
    gaps = []
    run = 0
    for o in obs:
        if not o:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    if run:
        gaps.append(run)
    g = np.asarray(gaps) * dt_min
    return {"days": len(s5) * dt_min / 1440, "coverage": float(obs.mean()) if len(obs) else 0.0,
            "n_gaps_ge_30min": int((g >= 30).sum()), "longest_gap_h": float(g.max() / 60) if len(g) else 0.0}


@dataclass
class Participant:
    participant_id: str
    status: str
    hba1c: float
    cgm: pd.Series                      # regular 5-min grid, NaN = unobserved
    report: dict[str, float] = field(default_factory=dict)


def load_participant(root: Path, pid: str, bio_row: pd.Series, sensor: str, dt_min: int,
                     bridged_gap_min: int) -> Participant | None:
    path = root / pid / f"{pid}.csv"
    if not path.is_file():
        return None
    raw, n_out_of_range = read_sensor(path, sensor)
    masked, n_bridged = mask_bridged_gaps(raw, bridged_gap_min) if len(raw) else (raw, 0)
    s5 = resample_5min(masked, dt_min)
    rep = {**gap_report(s5, dt_min), "out_of_range_readings": n_out_of_range, "bridged_minutes_masked": n_bridged}
    return Participant(pid, bio_row.status, float(bio_row.hba1c), s5, rep)


def load_cohort(data_dir: str | Path, sensor: str = DEFAULT_SENSOR, statuses: tuple[str, ...] = STATUSES,
                dt_min: int = 5, bridged_gap_min: int = BRIDGED_GAP_MIN) -> list[Participant]:
    """Every participant in `statuses` that has a CSV and at least one reading of `sensor`."""
    root = find_root(data_dir)
    bio = load_bio(root)
    out = []
    for pid, row in bio.iterrows():
        if row.status not in statuses:
            continue
        p = load_participant(root, str(pid), row, sensor, dt_min, bridged_gap_min)
        if p is not None and p.cgm.notna().any():
            out.append(p)
    return out


def cohort_table(cohort: list[Participant]) -> pd.DataFrame:
    return pd.DataFrame([{"participant_id": p.participant_id, "status": p.status, "hba1c": p.hba1c, **p.report}
                         for p in cohort])


def build_windows(cohort: list[Participant], cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Windows + labels (label_windows) and handcrafted features for every participant.

    Each participant has its own time grid, so they are windowed one at a time. Only the CGM channel is
    observed; steps / heart rate / sleep are NaN, so only the "cgm" and "time" feature groups are usable.
    Returns (windows, features) with aligned rows; windows carry patient_id and status.
    """
    dt = cfg["sampling"]["interval_minutes"]
    win_parts, feat_parts = [], []
    for i, p in enumerate(cohort):
        if len(p.cgm) == 0:
            continue
        ts = pd.DatetimeIndex(p.cgm.index)
        if (ts[1:] - ts[:-1] != pd.Timedelta(minutes=dt)).any():
            raise ValueError(f"{p.participant_id}: series is not on a regular {dt}-min grid")
        cgm = p.cgm.to_numpy(np.float32)[None, :]
        w = label_windows(cgm, ts, cfg)
        if w.empty:
            continue
        series = np.full((1, len(ts), len(SERIES_CHANNELS)), np.nan, np.float32)
        series[0, :, 0] = cgm[0]
        f = build_feature_table(series, w, ts, cfg)
        w = w.assign(patient_idx=i, patient_id=p.participant_id, status=p.status,
                     timestamp=ts[w.anchor_idx.to_numpy()])
        win_parts.append(w)
        feat_parts.append(f)
    if not win_parts:
        raise ValueError("no valid windows in the cohort")
    return pd.concat(win_parts, ignore_index=True), pd.concat(feat_parts, ignore_index=True)
