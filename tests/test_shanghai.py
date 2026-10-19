"""ShanghaiT2DM loader tests on a tiny made-up fixture in the dataset's file format (no real data)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("openpyxl")

from src.data.shanghai import dose_features, load_cohort, load_period, map_static  # noqa: E402

SUMMARY_COLUMNS = [
    "Patient Number", "Gender (Female=1, Male=2)", "Age (years)", "Height (m)", "Weight (kg)", "BMI (kg/m2)",
    "Smoking History (pack year)", "Alcohol Drinking History (drinker/non-drinker)", "Type of Diabetes",
    "Duration of diabetes (years)", "Acute Diabetic Complications", "Diabetic Macrovascular  Complications",
    "Diabetic Microvascular Complications", "Comorbidities", "Hypoglycemic Agents", "Other Agents",
    "Fasting Plasma Glucose (mg/dl)", "2-hour Postprandial Plasma Glucose (mg/dl)", "Fasting C-peptide (nmol/L)",
    "2-hour Postprandial C-peptide (nmol/L)", "Fasting Insulin (pmol/L)", "2-hour Postprandial insulin (pmol/L)",
    "HbA1c (mmol/mol)", "Glycated Albumin (%)", "Total Cholesterol (mmol/L)", "Triglyceride (mmol/L)",
    "High-Density Lipoprotein Cholesterol (mmol/L)", "Low-Density Lipoprotein Cholesterol (mmol/L)",
    "Creatinine (umol/L)", "Estimated Glomerular Filtration Rate  (ml/min/1.73m2) ", "Uric Acid (mmol/L)",
    "Blood Urea Nitrogen (mmol/L)", "Hypoglycemia (yes/no)"]
FILE_COLUMNS = ["Date", "CGM (mg / dl)", "CBG (mg / dl)", "Blood Ketone (mmol / L)", "Dietary intake", "饮食",
                "Insulin dose - s.c.", "Non-insulin hypoglycemic agents", "CSII - bolus insulin (Novolin R, IU)",
                "CSII - basal insulin (Novolin R, IU / H)", "Insulin dose - i.v."]
T0 = pd.Timestamp("2021-03-01 08:03:00")


def _summary_row(pid: str, sex: int, agents: str, comorb: str, macro: str, hypo: str) -> dict:
    row = {c: 1.0 for c in SUMMARY_COLUMNS}
    row.update({"Patient Number": pid, "Gender (Female=1, Male=2)": sex, "Age (years)": 60, "Weight (kg)": 70.0,
                "BMI (kg/m2)": 25.0, "Duration of diabetes (years)": 10.0, "Type of Diabetes": "T2DM",
                "Acute Diabetic Complications": "none", "Diabetic Macrovascular  Complications": macro,
                "Diabetic Microvascular Complications": "neuropathy", "Comorbidities": comorb,
                "Hypoglycemic Agents": agents, "Other Agents": "none", "HbA1c (mmol/mol)": 53.0,
                "Low-Density Lipoprotein Cholesterol (mmol/L)": 2.0, "Triglyceride (mmol/L)": 1.0,
                "Estimated Glomerular Filtration Rate  (ml/min/1.73m2) ": 90, "Hypoglycemia (yes/no)": hypo,
                "Alcohol Drinking History (drinker/non-drinker)": "non-drinker"})
    return row


def _period_frame(n: int, basal_header: str | None = None) -> pd.DataFrame:
    """15-min readings: a low at reading 40-43, one reading below the sensor range, a 45-min gap at 60-62."""
    t = T0 + pd.to_timedelta(np.arange(n) * 15, unit="min")
    g = 120 + 30 * np.sin(np.arange(n) / 6.0)
    g[40:44] = 60.0
    g[50] = 35.0                                              # below 40: clipped, not dropped
    df = pd.DataFrame({c: np.nan for c in FILE_COLUMNS}, index=range(n)).astype(object)
    df["Date"], df["CGM (mg / dl)"] = t, g
    df.loc[10, "Insulin dose - s.c."] = "Humulin 70/30, 8 IU"        # 8 units, not 70 or 30
    df.loc[20, "Insulin dose - s.c."] = "insulin aspart 70/30, 6 IU, Novolin R 4 IU"
    df.loc[30, "Non-insulin hypoglycemic agents"] = "gliclazide 60 mg, metformin 0.5 g"
    df.loc[32, "CSII - bolus insulin (Novolin R, IU)"] = 5
    df.loc[12, "Dietary intake"] = "rice 100 g"
    df.loc[25, "Dietary intake"] = "data not available"
    df = df.drop(index=[61, 62])                                     # 45-min gap between readings 60 and 63
    if basal_header:
        df = df.rename(columns={"CSII - basal insulin (Novolin R, IU / H)": basal_header})
    return df


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "2022.11 数据"                                  # the real top folder name is not ASCII
    (root / "Shanghai_T2DM").mkdir(parents=True)
    pd.DataFrame([_summary_row("2001_0_20210301", 2, "metformin, Humulin 70/30", "hypertension, hyperlipidemia",
                               "coronary heart disease", "yes"),
                  _summary_row("2001_1_20210320", 2, "gliclazide", "none", "none", "no"),
                  _summary_row("2002_0_20210301", 1, "insulin degludec, acarbose", "none", "none", "no")],
                 columns=SUMMARY_COLUMNS).to_excel(root / "Shanghai_T2DM_Summary.xlsx", index=False)
    _period_frame(200).to_excel(root / "Shanghai_T2DM" / "2001_0_20210301.xlsx", index=False)
    _period_frame(200, basal_header="胰岛素泵基础量 (Novolin R, IU / H)").to_excel(
        root / "Shanghai_T2DM" / "2001_1_20210320.xlsx", index=False)
    _period_frame(200).to_excel(root / "Shanghai_T2DM" / "2002_0_20210301.xlsx", index=False)
    (root / "Shanghai_T2DM" / "~$2002_0_20210301.xlsx").write_bytes(b"lock file")   # shipped in the real zip
    return tmp_path


def test_static_mapping_units_and_flags(data_dir: Path) -> None:
    periods, static, _ = load_cohort(data_dir)
    assert [p.period_id for p in periods] == ["2001_0_20210301", "2001_1_20210320", "2002_0_20210301"]
    assert [p.patient_id for p in periods] == ["2001", "2001", "2002"]             # two periods, one patient
    a = static.loc["2001_0_20210301"]
    assert a.hba1c == pytest.approx(53 / 10.929 + 2.15)                             # 53 mmol/mol = 7.0 %
    assert a.ldl_mgdl == pytest.approx(2.0 * 38.67) and a.triglycerides_mgdl == pytest.approx(88.57)
    assert (a.sex_male, a.on_metformin, a.on_insulin, a.insulin_premix, a.on_sulfonylurea) == (1, 1, 1, 1, 0)
    assert (a.hypertension, a.dyslipidemia, a.coronary_artery_disease, a.neuropathy) == (1, 1, 1, 1)
    assert np.isnan(a.prior_severe_hypo_12m)                     # the ambiguous Hypoglycemia field is not used
    b, c = static.loc["2001_1_20210320"], static.loc["2002_0_20210301"]
    assert (b.on_sulfonylurea, b.on_insulin) == (1, 0)
    assert (c.on_insulin, c.insulin_basal, c.insulin_premix, c.on_acarbose_like, c.sex_male) == (1, 1, 0, 1, 0)


def test_period_grid_interpolation_clipping_and_doses(data_dir: Path) -> None:
    path = next((data_dir / "2022.11 数据" / "Shanghai_T2DM").glob("2001_0*.xlsx"))
    cgm, doses, rep = load_period(path, 5)
    assert (cgm.index[1] - cgm.index[0]) == pd.Timedelta(minutes=5) and cgm.index[0] == T0
    assert cgm.iloc[3] == pytest.approx(120 + 30 * np.sin(1 / 6.0))             # readings land on the grid
    gap = cgm[T0 + pd.Timedelta(minutes=15 * 60): T0 + pd.Timedelta(minutes=15 * 63)]
    assert gap.iloc[1:-1].isna().all() and gap.iloc[[0, -1]].notna().all()      # > 30 min: not bridged
    assert cgm.min() == 40.0 and rep["clipped_readings"] == 1
    assert rep["readings_below_70"] == 5                                          # 4 lows + the clipped reading
    assert doses.insulin_units.sum() == pytest.approx(8 + 10 + 5)
    assert doses.secretagogue.sum() == 1 and doses.oral.sum() == 1 and doses.pump_or_iv.sum() == 1
    assert doses.meal.sum() == 1                                                  # "data not available" ignored


def test_chinese_header_falls_back_to_position(data_dir: Path) -> None:
    path = next((data_dir / "2022.11 数据" / "Shanghai_T2DM").glob("2001_1*.xlsx"))
    _, doses, _ = load_period(path, 5)
    assert doses.insulin_units.sum() == pytest.approx(23)


def test_dose_features_never_look_ahead() -> None:
    grid = pd.date_range(T0, periods=200, freq="5min")
    d = pd.DataFrame(0.0, index=grid, columns=["insulin_units", "secretagogue", "oral", "pump_or_iv", "meal"])
    d.iloc[100, 0] = 8.0
    f = dose_features(d, np.array([99, 100, 123, 124, 171, 172]), 5)
    assert list(f.insulin_units_2h) == [0, 8, 8, 0, 0, 0]          # 2 h = 24 steps: in at 100, out at 124
    assert list(f.insulin_units_6h) == [0, 8, 8, 8, 8, 0]          # 6 h = 72 steps: out at 172


def test_map_static_has_project_features_first(data_dir: Path) -> None:
    from src.data.ehr_features import STATIC_FEATURES
    _, static, summary = load_cohort(data_dir)
    assert list(static.columns[:len(STATIC_FEATURES)]) == list(STATIC_FEATURES)
    assert map_static(summary).equals(static)


def test_cold_start_features_forget_older_cgm(data_dir: Path) -> None:
    from src.config import load_config
    from src.eval.external_shanghai import CGM, build_windows
    from src.eval.external_shanghai_coldstart import short_history_features
    cfg = load_config()
    periods, static, _ = load_cohort(data_dir)
    w, full = build_windows(periods, static, cfg)
    assert len(w) > 0 and set(w.patient_id) == {"2001", "2002"}
    six = short_history_features(periods, w, 6.0, cfg)
    assert np.allclose(six.to_numpy(float), full[list(CGM)].to_numpy(float), equal_nan=True)
    one, zero = short_history_features(periods, w, 1.0, cfg), short_history_features(periods, w, 0.0, cfg)
    assert np.allclose(one.cgm_last, full.cgm_last) and not np.allclose(one.cgm_min, full.cgm_min, equal_nan=True)
    assert zero.cgm_last.isna().all() and zero.cgm_mean.isna().all()
    assert np.allclose(zero.hour, full.hour)                      # time of day is all that is left
