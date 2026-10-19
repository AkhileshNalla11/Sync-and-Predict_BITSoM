"""Loader for ShanghaiT2DM (Zhao et al., Scientific Data 2023), used only for the real linked-data check.

    https://doi.org/10.6084/m9.figshare.c.6310860   (CC BY 4.0; never committed to this repo)

100 adults with Type 2 diabetes from Shanghai, 109 recording periods of 2-14 days. Unlike CGMacros, each
period's CGM comes with the SAME patient's clinical record (summary sheet: demographics, HbA1c, eGFR,
lipids, complications, comorbidities, drug names) and a time-stamped log of meals, insulin and oral drugs.
That is the real counterpart of this project's synthetic EHR + CGM linkage.

Expected layout after unzipping diabetes_datasets.zip (the top folder name is not ASCII and may vary):

    data/external/shanghai/<folder>/Shanghai_T2DM_Summary.xlsx      one row per period ("2001_1_20201117")
    data/external/shanghai/<folder>/Shanghai_T2DM/<period>.xls[x]     15-min rows: Date, CGM, doses, diet

Choices (documented in docs/external_shanghai.md):
  * CGM every 15 min is placed on our 5-min grid by linear interpolation between readings at most
    MAX_INTERP_MIN apart; longer gaps stay NaN. Linear interpolation cannot create a new high or low, so
    the labels match labels computed on the raw readings.
  * Readings outside 40-400 mg/dL are clipped to that range (as our synthetic sensor model does) rather
    than dropped, so that the lowest readings still count as lows; the number clipped is reported.
  * EHR: mapped onto this project's 24 static features where an equivalent exists (unit conversions
    below); the rest are NaN, which LightGBM handles natively. The summary's "Hypoglycemia (yes/no)" field
    is NOT used: it is unclear whether it describes history or this recording, so it could leak the outcome.
  * Logged doses become time-varying features (insulin units in the last 2 / 6 h, secretagogue or any
    oral agent in the last 6 h, insulin pump or IV insulin in the last 6 h, meal logged in the last 2 h).
The files are untrusted input: only the needed columns are read and every value is parsed as a number or
matched against fixed keyword lists. Nothing in them is executed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.ehr_features import STATIC_FEATURES

SUMMARY_NAME = "Shanghai_T2DM_Summary.xlsx"
SERIES_DIR = "Shanghai_T2DM"
SENSOR_RANGE_MGDL = (40.0, 400.0)
NATIVE_MIN = 15
MAX_INTERP_MIN = 30
HBA1C_MMOL_TO_PCT = (1 / 10.929, 2.15)     # NGSP % = IFCC mmol/mol / 10.929 + 2.15
LDL_MMOL_TO_MGDL = 38.67
TG_MMOL_TO_MGDL = 88.57

# Drug name keywords (lower case). Sulfonylureas only for `on_sulfonylurea`; glinides are counted with
# them as insulin secretagogues in the time-varying dose features, since both cause lows.
INSULIN_WORDS = ("insulin", "novolin", "humulin", "gansulin")
PREMIX_WORDS = ("70/30", "30r", "40r", "50/50")
BASAL_WORDS = ("degludec", "detemir", "glargine", "glarigine")
SULFONYLUREAS = ("gliclazide", "glimepiride", "gliquidone", "glipizide", "glibenclamide", "glyburide")
GLINIDES = ("repaglinide", "nateglinide")
EXTRA_DRUGS = {"on_acarbose_like": ("acarbose", "voglibose"), "on_dpp4": ("sitagliptin", "linagliptin", "saxagliptin"),
               "on_sglt2": ("dapagliflozin", "canagliflozin", "empagliflozin"), "on_tzd": ("pioglitazone",),
               "on_glp1": ("liraglutide", "dulaglutide", "semaglutide"), "on_glinide": GLINIDES}

# Extra clinical fields a real record has and our synthetic EHR does not ("full EHR" variant).
EXTRA_LABS = {"fasting_glucose_mgdl": "fasting plasma glucose", "pp2h_glucose_mgdl": "2-hour postprandial plasma glucose",
              "fasting_c_peptide": "fasting c-peptide", "pp2h_c_peptide": "2-hour postprandial c-peptide",
              "fasting_insulin": "fasting insulin", "pp2h_insulin": "2-hour postprandial insulin",
              "glycated_albumin": "glycated albumin", "creatinine": "creatinine", "uric_acid": "uric acid",
              "bun": "blood urea nitrogen", "smoking_pack_years": "smoking history"}
COMPLICATIONS = {"neuropathy": "neuropathy", "retinopathy": "retinopathy", "nephropathy": "nephropathy",
                 "peripheral_arterial_disease": "peripheral arterial disease"}
EXTRA_STATIC = tuple(EXTRA_LABS) + tuple(COMPLICATIONS) + tuple(EXTRA_DRUGS)
DOSE_FEATURES = ("insulin_units_2h", "insulin_units_6h", "secretagogue_6h", "oral_agent_6h",
                 "insulin_pump_or_iv_6h", "meal_logged_2h")

# Per-period file columns, matched on the lower-case header prefix. All 109 files share one column order,
# but a few headers differ (one names the pump basal rate in Chinese), so the position is the fallback.
COLS = {"date": "date", "cgm": "cgm", "diet": "dietary intake", "insulin_sc": "insulin dose - s.c",
        "oral": "non-insulin", "csii_bolus": "csii - bolus", "csii_basal": "csii - basal", "insulin_iv": "insulin dose - i.v"}
POSITION = {"date": 0, "cgm": 1, "diet": 4, "insulin_sc": 6, "oral": 7, "csii_bolus": 8, "csii_basal": 9, "insulin_iv": 10}
N_FILE_COLUMNS = 11


@dataclass
class Period:
    period_id: str                      # e.g. "2001_1_20201117"
    patient_id: str                     # e.g. "2001" (periods of one person share it)
    cgm: pd.Series                      # regular 5-min grid, NaN = unobserved
    doses: pd.DataFrame                 # same grid: insulin_units, secretagogue, oral, pump_or_iv, meal
    report: dict[str, float] = field(default_factory=dict)


def find_root(data_dir: str | Path) -> Path:
    """The folder that holds the summary sheet (the zip's top folder name varies with the unzip tool)."""
    d = Path(data_dir)
    for cand in [d, *sorted(p for p in d.iterdir() if p.is_dir())] if d.is_dir() else []:
        if (cand / SUMMARY_NAME).is_file() and (cand / SERIES_DIR).is_dir():
            return cand
    raise FileNotFoundError(f"{SUMMARY_NAME} + {SERIES_DIR}/ not found under {d} (see src/data/shanghai.py)")


def _has(text: Any, words: tuple[str, ...]) -> bool:
    t = str(text).lower()
    return any(w in t for w in words)


def _num(v: Any) -> float:
    return float(pd.to_numeric(v, errors="coerce"))


def _col(df: pd.DataFrame, prefix: str) -> str:
    for c in df.columns:
        if str(c).strip().lower().startswith(prefix):
            return c
    raise KeyError(f"no column starting with {prefix!r}")


def _file_col(df: pd.DataFrame, key: str) -> pd.Series:
    """A period-file column by header prefix, else by its fixed position."""
    try:
        return df[_col(df, COLS[key])]
    except KeyError:
        if df.shape[1] != N_FILE_COLUMNS:
            raise
        return df.iloc[:, POSITION[key]]


def load_summary(root: Path) -> pd.DataFrame:
    """Summary sheet with stripped column names, indexed by period id."""
    s = pd.read_excel(root / SUMMARY_NAME)
    s.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in s.columns]
    s["period_id"] = s[_col(s, "patient number")].astype(str).str.strip()
    return s.set_index("period_id")


def map_static(summary: pd.DataFrame) -> pd.DataFrame:
    """Our 24 static EHR features (unscaled, NaN where the record has no equivalent) + the extra fields."""
    s = summary
    agents = s[_col(s, "hypoglycemic agents")].astype(str)
    comorb = s[_col(s, "comorbidities")].astype(str)
    macro = s[_col(s, "diabetic macrovascular")].astype(str)
    micro = s[_col(s, "diabetic microvascular")].astype(str)
    out = pd.DataFrame(index=s.index)
    out["age"] = s[_col(s, "age")].map(_num)
    out["bmi"] = s[_col(s, "bmi")].map(_num)
    out["weight_kg"] = s[_col(s, "weight")].map(_num)
    out["diabetes_duration_years"] = s[_col(s, "duration of diabetes")].map(_num)
    a, b = HBA1C_MMOL_TO_PCT
    out["hba1c"] = s[_col(s, "hba1c")].map(_num) * a + b
    out["egfr"] = s[_col(s, "estimated glomerular")].map(_num)
    out["ldl_mgdl"] = s[_col(s, "low-density")].map(_num) * LDL_MMOL_TO_MGDL
    out["triglycerides_mgdl"] = s[_col(s, "triglyceride")].map(_num) * TG_MMOL_TO_MGDL
    for c in ("insulin_total_daily_units", "insulin_units_per_kg", "sulfonylurea_dose_mg", "metformin_dose_mg",
              "family_history_diabetes", "sulfonylurea_bd", "prior_severe_hypo_12m"):
        out[c] = np.nan                                         # no equivalent field (or not trusted, see above)
    out["sex_male"] = (s[_col(s, "gender")].map(_num) == 2).astype(float)
    out["hypertension"] = comorb.map(lambda v: _has(v, ("hypertension",))).astype(float)
    out["dyslipidemia"] = comorb.map(lambda v: _has(v, ("hyperlipidemia", "dyslipidemia"))).astype(float)
    out["coronary_artery_disease"] = macro.map(lambda v: _has(v, ("coronary",))).astype(float)
    out["on_metformin"] = agents.map(lambda v: _has(v, ("metformin",))).astype(float)
    out["on_sulfonylurea"] = agents.map(lambda v: _has(v, SULFONYLUREAS)).astype(float)
    out["on_insulin"] = agents.map(lambda v: _has(v, INSULIN_WORDS)).astype(float)
    out["insulin_premix"] = agents.map(lambda v: _has(v, PREMIX_WORDS)).astype(float)
    out["insulin_basal"] = agents.map(lambda v: _has(v, BASAL_WORDS)).astype(float)
    for name, key in EXTRA_LABS.items():
        out[name] = s[_col(s, key)].map(_num)
    for name, key in COMPLICATIONS.items():
        out[name] = (micro + " " + macro).map(lambda v, k=key: _has(v, (k,))).astype(float)
    for name, words in EXTRA_DRUGS.items():
        out[name] = agents.map(lambda v, w=words: _has(v, w)).astype(float)
    return out[list(STATIC_FEATURES) + list(EXTRA_STATIC)]


def _units(v: Any) -> float:
    """Insulin units in one log cell: a bare number, or every '<n> IU' in a text such as 'Humulin 70/30, 8 IU'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    x = pd.to_numeric(v, errors="coerce")
    if np.isfinite(x):
        return float(x)
    return float(sum(float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*iu", str(v).lower())))


def _present(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() != ""


def load_period(path: Path, dt_min: int) -> tuple[pd.Series, pd.DataFrame, dict[str, float]]:
    """CGM on a regular `dt_min` grid plus the dose / meal log on the same grid."""
    raw = pd.read_excel(path)
    t = pd.to_datetime(_file_col(raw, "date"), errors="coerce")
    g = pd.to_numeric(_file_col(raw, "cgm"), errors="coerce")
    keep = t.notna() & g.notna()
    lo, hi = SENSOR_RANGE_MGDL
    n_clipped = int(((g[keep] < lo) | (g[keep] > hi)).sum())
    obs = pd.Series(g[keep].clip(lo, hi).to_numpy(), index=t[keep]).sort_index()
    obs = obs[~obs.index.duplicated()]
    grid = pd.date_range(obs.index[0], obs.index[-1], freq=f"{dt_min}min")
    # interpolate only between readings at most MAX_INTERP_MIN apart
    s = obs.reindex(obs.index.union(grid)).interpolate(method="time", limit_area="inside")
    prev = pd.Series(obs.index, index=obs.index).reindex(s.index, method="ffill")
    nxt = pd.Series(obs.index, index=obs.index).reindex(s.index, method="bfill")
    gap = (nxt - prev).dt.total_seconds() / 60
    s[(gap > MAX_INTERP_MIN) & ~s.index.isin(obs.index)] = np.nan
    cgm = s.reindex(grid)

    acc = {c: np.zeros(len(grid)) for c in ("insulin_units", "secretagogue", "oral", "pump_or_iv", "meal")}
    ok = t.notna().to_numpy()
    slot = np.zeros(len(t), int)
    slot[ok] = np.clip(np.round((t[ok] - grid[0]).dt.total_seconds().to_numpy() / 60 / dt_min), 0, len(grid) - 1)

    def put(col: str, values: pd.Series) -> None:
        v = values.to_numpy(float)
        m = ok & (v > 0)
        np.add.at(acc[col], slot[m], v[m])

    sc, iv = _file_col(raw, "insulin_sc"), _file_col(raw, "insulin_iv")
    bolus, basal = _file_col(raw, "csii_bolus"), _file_col(raw, "csii_basal")
    oral, diet = _file_col(raw, "oral"), _file_col(raw, "diet")
    put("insulin_units", sc.map(_units) + bolus.map(_units))
    put("secretagogue", oral.map(lambda v: float(_has(v, SULFONYLUREAS + GLINIDES))))
    put("oral", oral.map(lambda v: float(_present(v))))
    put("pump_or_iv", bolus.map(lambda v: float(_present(v))) + basal.map(lambda v: float(_present(v)))
        + iv.map(lambda v: float(_present(v))))
    put("meal", diet.map(lambda v: float(_present(v) and "not available" not in str(v).lower())))
    doses = pd.DataFrame(acc, index=grid)
    rep = {"days": len(grid) * dt_min / 1440, "readings": int(len(obs)), "coverage": float(cgm.notna().mean()),
           "clipped_readings": n_clipped, "readings_below_70": int((obs < 70).sum())}
    return cgm, doses, rep


def load_cohort(data_dir: str | Path, dt_min: int = 5) -> tuple[list[Period], pd.DataFrame, pd.DataFrame]:
    """(periods, static features per period, raw summary). Periods without a summary row are skipped."""
    root = find_root(data_dir)
    summary = load_summary(root)
    static = map_static(summary)
    periods = []
    for path in sorted((root / SERIES_DIR).glob("*.xls*")):
        if path.name.startswith("~$"):                        # Excel lock files shipped in the zip
            continue
        pid = path.stem
        if pid not in static.index:
            continue
        cgm, doses, rep = load_period(path, dt_min)
        periods.append(Period(pid, pid.split("_")[0], cgm, doses, rep))
    if not periods:
        raise ValueError(f"no readable periods under {root / SERIES_DIR}")
    return periods, static, summary


def dose_features(doses: pd.DataFrame, anchors: np.ndarray, dt_min: int) -> pd.DataFrame:
    """Time-varying log features at each anchor (inclusive of the anchor step, never after it)."""
    def window_sum(col: str, hours: float) -> np.ndarray:
        c = np.r_[0.0, np.cumsum(doses[col].to_numpy(float))]
        k = int(hours * 60 // dt_min)
        hi = anchors + 1
        return c[hi] - c[np.maximum(hi - k, 0)]
    return pd.DataFrame({
        "insulin_units_2h": window_sum("insulin_units", 2), "insulin_units_6h": window_sum("insulin_units", 6),
        "secretagogue_6h": (window_sum("secretagogue", 6) > 0).astype(float),
        "oral_agent_6h": (window_sum("oral", 6) > 0).astype(float),
        "insulin_pump_or_iv_6h": (window_sum("pump_or_iv", 6) > 0).astype(float),
        "meal_logged_2h": (window_sum("meal", 2) > 0).astype(float),
    })
