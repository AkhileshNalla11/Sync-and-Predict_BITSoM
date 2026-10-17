"""Synthetic EHR cohort generation (Stage 1) and static feature encoding (Stage 2).

Every record here is SYNTHETIC. Distributions are loosely inspired by published
descriptions of Indian Type 2 Diabetes cohorts (high carbohydrate diets, onset
in the 40s-50s, BMI mid-20s, frequent hypertension and family history), but no
real patient data was used and none of the numbers should be read as clinical
epidemiology.

BY DESIGN, EHR traits drive the wearable simulator (see synthetic_gen.py):
  * higher HbA1c / BMI          -> higher insulin resistance -> larger post-meal spikes
  * insulin or sulfonylurea use -> drug-driven glucose lowering -> hypoglycaemia risk
  * low eGFR                    -> slower drug clearance -> longer, deeper lows
  * long diabetes duration      -> weaker counter-regulation -> deeper lows
  * prior severe hypoglycaemia  -> correlated with a hidden drug-sensitivity trait
This guarantees that the static stream carries real signal for the fusion model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMENS = ("none", "metformin", "metformin+sulfonylurea", "sulfonylurea",
            "insulin", "insulin+metformin", "insulin+metformin+sulfonylurea")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _regimen_label(met: bool, su: bool, ins: bool) -> str:
    parts = [name for name, on in (("insulin", ins), ("metformin", met), ("sulfonylurea", su)) if on]
    return "+".join(parts) if parts else "none"


def _icd10_codes(row: pd.Series) -> str:
    """Plausible ICD-10 code list for display; derived from the other fields."""
    codes = ["E11.65" if row.hba1c >= 9.0 else "E11.9"]
    if row.egfr < 60:
        codes += ["E11.22", "N18.3" if row.egfr >= 30 else "N18.4"]
    if row.prior_severe_hypo_12m:
        codes.append("E11.649")
    if row.hypertension:
        codes.append("I10")
    if row.dyslipidemia:
        codes.append("E78.5")
    if row.coronary_artery_disease:
        codes.append("I25.10")
    return ";".join(codes)


def generate_cohort(n_patients: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the synthetic EHR table plus hidden latent traits.

    Returns:
        ehr:    observable EHR fields, one row per patient (what a clinician/model sees).
        latent: hidden traits used only by the simulator (never a model input).
    """
    n = n_patients
    ids = np.array([f"P{i:04d}" for i in range(1, n + 1)])

    age = np.clip(rng.normal(55, 10, n), 30, 82).round()
    sex = np.where(rng.random(n) < 0.55, "M", "F")
    height_cm = np.where(sex == "M", rng.normal(167, 6.5, n), rng.normal(154, 6.0, n))
    bmi = np.clip(rng.normal(26.5, 4.0, n), 18.5, 42.0)
    weight_kg = bmi * (height_cm / 100) ** 2
    duration = np.clip(rng.gamma(2.2, 3.2, n) * (0.6 + 0.4 * (age - 30) / 50), 0.3, age - 22)

    hba1c = np.clip(7.3 + 0.06 * (duration - 7) + 0.03 * (bmi - 26.5) + rng.normal(0, 0.95, n), 5.7, 12.5)
    egfr = np.clip(105 - 0.75 * (age - 40) - 0.5 * duration + rng.normal(0, 13, n), 12, 125)

    hypertension = rng.random(n) < _sigmoid(-0.4 + 0.05 * (age - 55) + 0.10 * (bmi - 26.5))
    dyslipidemia = rng.random(n) < 0.55
    cad = rng.random(n) < _sigmoid(-2.6 + 0.06 * (age - 55) + 0.5 * hypertension + 0.3 * (sex == "M"))
    family_history = rng.random(n) < 0.60
    ldl = np.clip(rng.normal(105, 30, n), 40, 220)
    triglycerides = np.clip(rng.lognormal(np.log(150) + 0.02 * (bmi - 26.5), 0.35, n), 50, 600)

    # Treatment intensification follows duration and control, as in routine care.
    on_insulin = rng.random(n) < _sigmoid(-1.4 + 0.18 * (duration - 7) + 0.9 * (hba1c - 8.0) + 0.8 * (egfr < 45))
    on_su = rng.random(n) < _sigmoid(-0.6 + 0.08 * (duration - 5) + 0.5 * (hba1c - 7.5) - 1.0 * on_insulin)
    on_met = (egfr >= 30) & (rng.random(n) < 0.90)          # metformin contraindicated at eGFR < 30
    diet_only = (duration < 2) & (hba1c < 7.0) & (rng.random(n) < 0.6)
    on_insulin &= ~diet_only
    on_su &= ~diet_only
    on_met &= ~diet_only

    insulin_regimen = np.where(on_insulin, np.where(rng.random(n) < 0.7, "premix_bid", "basal"), "")
    # Doses are titrated to control: tighter HbA1c -> lower doses (de-intensification).
    u_per_kg = np.clip(rng.normal(0.45, 0.10, n) + 0.06 * (hba1c - 8.0), 0.2, 0.9)
    u_per_kg = np.where(insulin_regimen == "basal", u_per_kg * 0.55, u_per_kg)
    insulin_units = np.where(on_insulin, np.round(weight_kg * u_per_kg), 0.0)
    # Glimepiride-equivalent total daily dose; twice-daily (BD) split is common in Indian practice.
    su_dose_mg = np.where(on_su, np.clip(np.round(1.8 + 0.7 * (hba1c - 7.5) + rng.normal(0, 0.7, n)), 1, 4), 0)
    su_frequency = np.where(on_su, np.where(rng.random(n) < 0.5, "BD", "OD"), "")
    metformin_dose_mg = np.where(on_met, rng.choice([1000, 1500, 2000], n, p=[0.4, 0.35, 0.25]), 0)

    # Hidden trait: how strongly this patient's glucose responds to insulin / SU
    # (titration mismatch, erratic absorption...). Prior severe hypo correlates with it.
    # Capped: in routine care a patient this over-treated would be de-escalated.
    drug_sens = np.clip(rng.lognormal(0.0, 0.5, n), 0.4, 1.6)
    on_hypo_drug = on_insulin | on_su
    prior_hypo = on_hypo_drug & (rng.random(n) < _sigmoid(
        -2.2 + 2.0 * np.log(drug_sens) + 0.8 * (egfr < 45) + 0.03 * (age - 60)))

    ehr = pd.DataFrame({
        "patient_id": ids,
        "age": age.astype(int),
        "sex": sex,
        "height_cm": height_cm.round(1),
        "weight_kg": weight_kg.round(1),
        "bmi": bmi.round(1),
        "diabetes_duration_years": duration.round(1),
        "hba1c": hba1c.round(1),
        "egfr": egfr.round(0),
        "ldl_mgdl": ldl.round(0),
        "triglycerides_mgdl": triglycerides.round(0),
        "hypertension": hypertension,
        "dyslipidemia": dyslipidemia,
        "coronary_artery_disease": cad,
        "family_history_diabetes": family_history,
        "on_metformin": on_met,
        "on_sulfonylurea": on_su,
        "on_insulin": on_insulin,
        "regimen": [_regimen_label(m, s, i) for m, s, i in zip(on_met, on_su, on_insulin)],
        "insulin_regimen": insulin_regimen,
        "insulin_total_daily_units": insulin_units,
        "sulfonylurea_dose_mg": su_dose_mg,
        "sulfonylurea_frequency": su_frequency,
        "metformin_dose_mg": metformin_dose_mg,
        "prior_severe_hypo_12m": prior_hypo,
    })
    ehr["diagnoses_icd10"] = ehr.apply(_icd10_codes, axis=1)

    latent = pd.DataFrame({
        "patient_id": ids,
        "drug_sensitivity": drug_sens,
        "ir_noise": rng.normal(0, 0.12, n),
    })
    return ehr, latent


# --------------------------------------------------------------------------- static features (Stage 2)
# Free-text / derived columns (patient_id, regimen, diagnoses_icd10, height) are deliberately excluded:
# the codes are a deterministic re-statement of the fields below.
NUMERIC_FEATURES = ("age", "bmi", "weight_kg", "diabetes_duration_years", "hba1c", "egfr", "ldl_mgdl",
                    "triglycerides_mgdl", "insulin_total_daily_units", "insulin_units_per_kg",
                    "sulfonylurea_dose_mg", "metformin_dose_mg")
BINARY_FEATURES = ("sex_male", "hypertension", "dyslipidemia", "coronary_artery_disease",
                   "family_history_diabetes", "on_metformin", "on_sulfonylurea", "on_insulin",
                   "insulin_premix", "insulin_basal", "sulfonylurea_bd", "prior_severe_hypo_12m")
STATIC_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES


def encode_static(ehr: pd.DataFrame) -> pd.DataFrame:
    """One numeric row per patient (unscaled), indexed by patient_id, columns = STATIC_FEATURES."""
    df = pd.DataFrame(index=ehr.patient_id.to_numpy())
    for col in NUMERIC_FEATURES:
        if col == "insulin_units_per_kg":
            df[col] = (ehr.insulin_total_daily_units / ehr.weight_kg).to_numpy()
        else:
            df[col] = ehr[col].to_numpy(dtype=float)
    df["sex_male"] = (ehr.sex == "M").to_numpy()
    df["insulin_premix"] = (ehr.insulin_regimen == "premix_bid").to_numpy()
    df["insulin_basal"] = (ehr.insulin_regimen == "basal").to_numpy()
    df["sulfonylurea_bd"] = (ehr.sulfonylurea_frequency == "BD").to_numpy()
    for col in BINARY_FEATURES:
        if col not in df:
            df[col] = ehr[col].to_numpy()
    return df[list(STATIC_FEATURES)].astype(float)


class StaticEncoder:
    """Standardises numeric EHR features using statistics from the TRAINING patients only.

    Binary features pass through as 0/1. Parameters serialise to JSON for the twin/dashboard.
    """

    def __init__(self) -> None:
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(self, encoded: pd.DataFrame) -> "StaticEncoder":
        for col in NUMERIC_FEATURES:
            self.mean_[col] = float(encoded[col].mean())
            std = float(encoded[col].std(ddof=0))
            self.std_[col] = std if std > 1e-9 else 1.0
        return self

    def transform(self, encoded: pd.DataFrame) -> pd.DataFrame:
        if not self.mean_:
            raise RuntimeError("StaticEncoder.transform called before fit")
        out = encoded.copy()
        for col in NUMERIC_FEATURES:
            out[col] = (encoded[col] - self.mean_[col]) / self.std_[col]
        return out

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"mean": self.mean_, "std": self.std_}

    @classmethod
    def from_dict(cls, d: dict[str, dict[str, float]]) -> "StaticEncoder":
        enc = cls()
        enc.mean_, enc.std_ = dict(d["mean"]), dict(d["std"])
        return enc
