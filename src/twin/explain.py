"""Per-prediction risk drivers for the twin.

Method: exact TreeSHAP on the LightGBM predictor (LightGBM's built-in `pred_contrib`, the same
algorithm `shap.TreeExplainer` implements; a test checks they agree). Contributions are in log-odds:
positive pushes the predicted risk up, negative pushes it down, and they sum with the base value to
the model's logit. Drivers explain the MODEL, not the patient's physiology: they say what the model
reacted to, not what caused the glucose excursion.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.twin.state import PatientTwin, TwinContext

Fmt = Callable[[float], str]


def _num(unit: str = "", digits: int = 0) -> Fmt:
    return lambda v: "n/a" if not np.isfinite(v) else f"{v:.{digits}f}{unit}"


def _minutes(v: float) -> str:
    return "none in last 6 h" if v >= 400 else f"{v:.0f} min ago"


def _yes(v: float) -> str:
    return "yes" if v >= 0.5 else "no"


def _hour(v: float) -> str:
    return f"{int(v):02d}:{int(round((v % 1) * 60)) % 60:02d}"


# feature -> (label, group, value formatter)
DISPLAY: dict[str, tuple[str, str, Fmt]] = {
    "cgm_last": ("Current glucose", "CGM", _num(" mg/dL")),
    "cgm_minutes_since_obs": ("Time since last CGM reading", "CGM", _num(" min")),
    "cgm_mean": ("Mean glucose, 6 h", "CGM", _num(" mg/dL")),
    "cgm_std": ("Glucose variability (SD), 6 h", "CGM", _num(" mg/dL")),
    "cgm_cv": ("Glucose variability (CV), 6 h", "CGM", lambda v: f"{100 * v:.0f}%"),
    "cgm_min": ("Lowest glucose, 6 h", "CGM", _num(" mg/dL")),
    "cgm_max": ("Highest glucose, 6 h", "CGM", _num(" mg/dL")),
    "cgm_mean_1h": ("Mean glucose, last hour", "CGM", _num(" mg/dL")),
    "cgm_min_1h": ("Lowest glucose, last hour", "CGM", _num(" mg/dL")),
    "cgm_max_1h": ("Highest glucose, last hour", "CGM", _num(" mg/dL")),
    "cgm_slope_30m": ("Glucose trend, last 30 min", "CGM", lambda v: "n/a" if not np.isfinite(v) else f"{v:+.1f} mg/dL/min"),
    "cgm_slope_60m": ("Glucose trend, last 60 min", "CGM", lambda v: "n/a" if not np.isfinite(v) else f"{v:+.1f} mg/dL/min"),
    "cgm_accel": ("Change in trend", "CGM", lambda v: "n/a" if not np.isfinite(v) else f"{v:+.2f} mg/dL/min"),
    "cgm_delta_15m": ("Change over last 15 min", "CGM", lambda v: f"{v:+.0f} mg/dL"),
    "cgm_frac_above_180": ("Time above 180, 6 h", "CGM", lambda v: f"{100 * v:.0f}%"),
    "cgm_frac_below_70": ("Time below 70, 6 h", "CGM", lambda v: f"{100 * v:.0f}%"),
    "min_since_above_180": ("Last reading above 180", "CGM", _minutes),
    "min_since_below_70": ("Last reading below 70", "CGM", _minutes),
    "min_since_meal_rise": ("Last meal-like glucose rise", "CGM", _minutes),
    "max_rise_3h": ("Largest 30-min rise, 3 h", "CGM", _num(" mg/dL")),
    "cgm_coverage": ("CGM data coverage, 6 h", "CGM", lambda v: f"{100 * v:.0f}%"),
    "hour": ("Time of day", "Time", _hour),
    "tod_sin": ("Time of day (cyclic)", "Time", _num("", 2)),
    "tod_cos": ("Time of day (cyclic)", "Time", _num("", 2)),
    "steps_30m": ("Steps, last 30 min", "Activity & sleep", _num()),
    "steps_1h": ("Steps, last hour", "Activity & sleep", _num()),
    "steps_3h": ("Steps, last 3 h", "Activity & sleep", _num()),
    "steps_max_1h": ("Peak 5-min steps, last hour", "Activity & sleep", _num()),
    "min_since_active": ("Last brisk activity", "Activity & sleep", _minutes),
    "wear_coverage": ("Wearable worn, 6 h", "Activity & sleep", lambda v: f"{100 * v:.0f}%"),
    "hr_last": ("Heart rate now", "Activity & sleep", _num(" bpm")),
    "hr_mean_30m": ("Heart rate, last 30 min", "Activity & sleep", _num(" bpm")),
    "hr_minus_6h_mean": ("Heart rate vs 6 h average", "Activity & sleep", lambda v: "n/a" if not np.isfinite(v) else f"{v:+.0f} bpm"),
    "asleep_now": ("Asleep now", "Activity & sleep", _yes),
    "asleep_minutes": ("Sleep in last 6 h", "Activity & sleep", _num(" min")),
    "awake_in_bed_minutes": ("Awake in bed, 6 h", "Activity & sleep", _num(" min")),
    "deep_frac": ("Deep-sleep share", "Activity & sleep", lambda v: f"{100 * v:.0f}%"),
    "age": ("Age", "EHR", _num(" y")),
    "bmi": ("BMI", "EHR", _num("", 1)),
    "weight_kg": ("Weight", "EHR", _num(" kg")),
    "diabetes_duration_years": ("Diabetes duration", "EHR", _num(" y", 1)),
    "hba1c": ("HbA1c", "EHR", _num("%", 1)),
    "egfr": ("eGFR (kidney function)", "EHR", _num("")),
    "ldl_mgdl": ("LDL cholesterol", "EHR", _num(" mg/dL")),
    "triglycerides_mgdl": ("Triglycerides", "EHR", _num(" mg/dL")),
    "insulin_total_daily_units": ("Insulin dose", "EHR", _num(" U/day")),
    "insulin_units_per_kg": ("Insulin dose per kg", "EHR", _num(" U/kg", 2)),
    "sulfonylurea_dose_mg": ("Sulfonylurea dose", "EHR", _num(" mg/day")),
    "metformin_dose_mg": ("Metformin dose", "EHR", _num(" mg/day")),
    "sex_male": ("Male", "EHR", _yes),
    "hypertension": ("Hypertension", "EHR", _yes),
    "dyslipidemia": ("Dyslipidaemia", "EHR", _yes),
    "coronary_artery_disease": ("Coronary artery disease", "EHR", _yes),
    "family_history_diabetes": ("Family history of diabetes", "EHR", _yes),
    "on_metformin": ("On metformin", "EHR", _yes),
    "on_sulfonylurea": ("On sulfonylurea", "EHR", _yes),
    "on_insulin": ("On insulin", "EHR", _yes),
    "insulin_premix": ("Premixed insulin", "EHR", _yes),
    "insulin_basal": ("Basal insulin", "EHR", _yes),
    "sulfonylurea_bd": ("Sulfonylurea twice daily", "EHR", _yes),
    "prior_severe_hypo_12m": ("Severe hypo in past 12 months", "EHR", _yes),
}
GROUPS = ("CGM", "Time", "Activity & sleep", "EHR")
TIME_PARTS = ("hour", "tod_sin", "tod_cos")


def contributions(ctx: TwinContext, X: pd.DataFrame, target: str) -> tuple[np.ndarray, float]:
    """TreeSHAP values (n_rows, n_features) in log-odds, plus the base value."""
    b = ctx.boosters[target]
    c = b.predict(X, num_iteration=b.best_iteration, pred_contrib=True)
    return c[:, :-1], float(c[0, -1])


def explain(ctx: TwinContext, twin: PatientTwin, target: str, top_k: int = 6) -> pd.DataFrame:
    """Top drivers of one twin's risk, ordered by |contribution|, with readable labels and values.

    Static features are shown with their raw (unscaled) values.
    """
    shap_vals, _ = contributions(ctx, twin.features, target)
    rows, time_contrib = [], 0.0
    for j, f in enumerate(ctx.feature_names):
        if f in TIME_PARTS:                       # one clock-time driver instead of hour / sin / cos
            time_contrib += float(shap_vals[0, j])
            continue
        label, group, fmt = DISPLAY.get(f, (f, "Other", _num("", 2)))
        raw = twin.static_raw[f] if f in twin.static_raw.index else twin.features[f].iloc[0]
        rows.append({"feature": f, "label": label, "group": group, "value": fmt(float(raw)),
                     "contribution": float(shap_vals[0, j])})
    if "hour" in ctx.feature_names:
        rows.append({"feature": "time_of_day", "label": "Time of day", "group": "Time",
                     "value": _hour(float(twin.features["hour"].iloc[0])), "contribution": time_contrib})
    df = pd.DataFrame(rows)
    df = df.reindex(df.contribution.abs().sort_values(ascending=False).index)
    df["direction"] = np.where(df.contribution > 0, "raises risk", "lowers risk")
    return df.head(top_k).reset_index(drop=True)


def group_contributions(ctx: TwinContext, twin: PatientTwin, target: str) -> pd.Series:
    """Summed contribution per data stream (CGM / time / activity & sleep / EHR)."""
    shap_vals, base = contributions(ctx, twin.features, target)
    groups = [DISPLAY.get(f, (f, "Other", None))[1] for f in ctx.feature_names]
    s = pd.Series(shap_vals[0], index=groups).groupby(level=0).sum()
    return s.reindex([g for g in GROUPS if g in s.index]).rename("log_odds")


def driver_sentences(drivers: pd.DataFrame, target: str) -> list[str]:
    """Short plain-language lines, e.g. 'Current glucose 82 mg/dL raises hypo risk'."""
    name = "hyperglycaemia" if target == "hyper" else "hypoglycaemia"
    return [f"{r.label}: {r.value} ({r.direction.replace('risk', name + ' risk')})" for r in drivers.itertuples()]


def explain_payload(ctx: TwinContext, twin: PatientTwin, top_k: int = 6) -> dict[str, Any]:
    """Everything the dashboard needs for both targets."""
    return {t: {"drivers": explain(ctx, twin, t, top_k), "groups": group_contributions(ctx, twin, t)}
            for t in ("hyper", "hypo")}
