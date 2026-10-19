"""FHIR R4 in and out: read a patient's record and sensor data from a Bundle, write risks as RiskAssessment.

This is the integration path a hospital system or an ABDM-connected app would use: it sends one FHIR Bundle
(the patient's record plus the last 6 hours of sensor readings) and gets back FHIR RiskAssessment resources.

Read (parse_bundle):
  * Patient                     gender, birthDate -> sex, age
  * Observation (LOINC)         CGM 99504-3 (glucose in interstitial fluid, mg/dL or mmol/L); HbA1c 4548-4 /
                                17856-6; eGFR 62238-1 / 33914-3 / 98979-8 / 48642-3 / 48643-1; LDL 13457-7 /
                                18262-6 / 2089-1; triglycerides 2571-8; BMI 39156-5; weight 29463-7 / 3141-9;
                                steps per interval 55423-8; heart rate 8867-4; sleep stage (local code system,
                                SLEEP_SYSTEM). Labs: the latest value wins.
  * Condition (ICD-10 or SNOMED) type 2 diabetes (onset -> duration), hypertension, dyslipidaemia, coronary
                                artery disease, hypoglycaemia in the last 12 months
  * FamilyMemberHistory         type 2 diabetes in a relative
  * MedicationStatement         metformin, sulfonylureas, insulin (premixed or basal), with dose and frequency.
                                Doses are kept only where this project's EHR has a matching unit: metformin mg,
                                glimepiride mg (other sulfonylureas: on/off only, no dose conversion), insulin U.
Write (to_bundle): the reverse, for exporting a synthetic patient as an example and for round-trip tests.
Anything missing stays NaN: the real-patient-trained model needs only CGM; the synthetic-trained model handles
missing EHR fields natively but was trained with all of them present.
SYNTHETIC DATA IN THE EXAMPLES. Not a medical device, not for clinical use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

LOINC = "http://loinc.org"
ICD10 = "http://hl7.org/fhir/sid/icd-10"
SNOMED = "http://snomed.info/sct"
SLEEP_SYSTEM = "https://github.com/AkhileshNalla11/glycaemic-digital-twin/fhir/sleep-stage"
MGDL_PER_MMOL = 18.016                      # glucose
CGM_CODES = {"99504-3"}
LABS = {"hba1c": {"4548-4", "17856-6"}, "egfr": {"62238-1", "33914-3", "98979-8", "48642-3", "48643-1"},
        "ldl_mgdl": {"13457-7", "18262-6", "2089-1"}, "triglycerides_mgdl": {"2571-8"}, "bmi": {"39156-5"},
        "weight_kg": {"29463-7", "3141-9"}}
LIPID_MMOL = {"ldl_mgdl": 38.67, "triglycerides_mgdl": 88.57}
STEPS, HEART_RATE = {"55423-8"}, {"8867-4"}
SLEEP_STAGES = ("awake", "light", "deep", "rem")
CONDITIONS = {  # field -> (ICD-10 prefixes, SNOMED codes)
    "t2d": (("E11",), {"44054006"}),
    "hypertension": (("I10", "I11", "I12", "I13", "I15"), {"38341003"}),
    "dyslipidemia": (("E78",), {"55822004", "370992007"}),
    "coronary_artery_disease": (("I20", "I21", "I22", "I23", "I24", "I25"), {"53741008"}),
    "hypoglycaemia": (("E16.0", "E16.1", "E16.2", "E11.64"), {"302866003"}),
}
SULFONYLUREAS = ("glimepiride", "gliclazide", "glipizide", "glibenclamide", "glyburide", "gliquidone")
PREMIX_WORDS = ("30/70", "70/30", "premix", "biphasic", "mix")
BASAL_WORDS = ("glargine", "degludec", "detemir", "basal", "nph", "isophane")
YEAR_DAYS = 365.25


@dataclass
class PatientRecord:
    patient_id: str
    profile: dict[str, Any]                        # this project's EHR fields (NaN = not in the record)
    readings: pd.DataFrame                         # timestamp, channel (cgm_mgdl / steps / heart_rate_bpm / sleep_stage), value
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- reading
def _codes(cc: dict[str, Any] | None) -> list[tuple[str, str]]:
    return [(c.get("system", ""), str(c.get("code", ""))) for c in (cc or {}).get("coding", [])]


def _has_code(cc: dict[str, Any] | None, icd: tuple[str, ...], snomed: set[str]) -> bool:
    for system, code in _codes(cc):
        if system == ICD10 and code.upper().startswith(icd):
            return True
        if system == SNOMED and code in snomed:
            return True
    return False


def _loinc(obs: dict[str, Any]) -> set[str]:
    return {code for system, code in _codes(obs.get("code")) if system == LOINC}


def _time(obs: dict[str, Any]) -> pd.Timestamp | None:
    t = obs.get("effectiveDateTime") or (obs.get("effectivePeriod") or {}).get("start") or obs.get("issued")
    return None if t is None else _naive(pd.Timestamp(t))


def _naive(t: pd.Timestamp) -> pd.Timestamp:
    """Keep the patient's local wall-clock time (time of day is a model input) and drop any UTC offset."""
    return t.tz_localize(None) if t.tzinfo is not None else t


def _quantity(obs: dict[str, Any]) -> tuple[float, str]:
    q = obs.get("valueQuantity") or {}
    v = q.get("value")
    return (float(v) if v is not None else math.nan), str(q.get("unit") or q.get("code") or "").lower()


def _med_name(ms: dict[str, Any]) -> str:
    cc = ms.get("medicationCodeableConcept") or {}
    parts = [cc.get("text", "")] + [c.get("display", "") for c in cc.get("coding", [])]
    return " ".join(p for p in parts if p).lower()


def _daily_dose(ms: dict[str, Any]) -> tuple[float, float, str]:
    """(dose per administration, administrations per day, unit) from the first dosage entry."""
    d = (ms.get("dosage") or [{}])[0]
    dr = (d.get("doseAndRate") or [{}])[0].get("doseQuantity") or {}
    rep = (d.get("timing") or {}).get("repeat") or {}
    freq = float(rep.get("frequency", 1)) / max(float(rep.get("period", 1)), 1e-9)
    if str(rep.get("periodUnit", "d")) == "h":
        freq *= 24
    v = dr.get("value")
    return (float(v) if v is not None else math.nan), freq, str(dr.get("unit") or dr.get("code") or "").lower()


def parse_bundle(bundle: dict[str, Any], now: pd.Timestamp | None = None) -> PatientRecord:
    """A FHIR Bundle (any type) -> this project's EHR profile + time-stamped sensor readings."""
    if bundle.get("resourceType") != "Bundle":
        raise ValueError("expected a FHIR Bundle")
    res = [e.get("resource") or {} for e in bundle.get("entry", [])]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in res:
        by_type.setdefault(r.get("resourceType", ""), []).append(r)
    patients = by_type.get("Patient", [])
    if len(patients) != 1:
        raise ValueError(f"expected exactly one Patient, found {len(patients)}")
    pat = patients[0]
    warnings: list[str] = []

    rows, labs = [], {}
    for obs in by_type.get("Observation", []):
        if obs.get("status") in ("entered-in-error", "cancelled"):
            continue
        codes, t = _loinc(obs), _time(obs)
        if t is None:
            continue
        local = {code for system, code in _codes(obs.get("code")) if system == SLEEP_SYSTEM}
        if codes & CGM_CODES:
            v, unit = _quantity(obs)
            rows.append((t, "cgm_mgdl", v * MGDL_PER_MMOL if "mmol" in unit else v))
        elif codes & STEPS:
            rows.append((t, "steps", _quantity(obs)[0]))
        elif codes & HEART_RATE:
            rows.append((t, "heart_rate_bpm", _quantity(obs)[0]))
        elif local or (obs.get("valueCodeableConcept") and any(s == SLEEP_SYSTEM for s, _ in _codes(obs["valueCodeableConcept"]))):
            stage = next((c for s, c in _codes(obs.get("valueCodeableConcept")) if s == SLEEP_SYSTEM), None)
            if stage in SLEEP_STAGES:
                rows.append((t, "sleep_stage", float(SLEEP_STAGES.index(stage))))
        else:
            for name, lc in LABS.items():
                if codes & lc:
                    v, unit = _quantity(obs)
                    if name in LIPID_MMOL and "mmol" in unit:
                        v *= LIPID_MMOL[name]
                    if name == "hba1c" and "mmol/mol" in unit:
                        v = v / 10.929 + 2.15
                    if name not in labs or t >= labs[name][0]:
                        labs[name] = (t, v)
    readings = pd.DataFrame(rows, columns=["timestamp", "channel", "value"]).sort_values("timestamp", kind="stable")
    if now is None:
        cgm_t = readings.timestamp[readings.channel == "cgm_mgdl"]
        now = cgm_t.max() if len(cgm_t) else pd.Timestamp.now()
    now = _naive(pd.Timestamp(now))

    prof: dict[str, Any] = {k: math.nan for k in ("age", "bmi", "weight_kg", "diabetes_duration_years", "hba1c", "egfr",
                                                   "ldl_mgdl", "triglycerides_mgdl", "insulin_total_daily_units",
                                                   "sulfonylurea_dose_mg", "metformin_dose_mg")}
    prof.update({k: v for k, (_, v) in labs.items()})
    gender = pat.get("gender")
    prof["sex"] = {"male": "M", "female": "F"}.get(gender, "")
    if pat.get("birthDate"):
        b = pd.Timestamp(pat["birthDate"])
        prof["age"] = now.year - b.year - ((now.month, now.day) < (b.month, b.day))

    flags = {k: False for k in CONDITIONS}
    for cond in by_type.get("Condition", []):
        status = _codes(cond.get("clinicalStatus"))
        if status and status[0][1] in ("inactive", "resolved", "remission"):
            continue
        for k, (icd, sn) in CONDITIONS.items():
            if _has_code(cond.get("code"), icd, sn):
                onset = cond.get("onsetDateTime")
                if k == "hypoglycaemia":
                    if onset and (now - _naive(pd.Timestamp(onset))).days > YEAR_DAYS:
                        continue
                if k == "t2d" and onset:
                    prof["diabetes_duration_years"] = (now - _naive(pd.Timestamp(onset))).total_seconds() / (YEAR_DAYS * 86400)
                flags[k] = True
    if not flags["t2d"]:
        warnings.append("no active type 2 diabetes Condition (E11 / 44054006) in the bundle")
    prof.update(hypertension=flags["hypertension"], dyslipidemia=flags["dyslipidemia"],
                coronary_artery_disease=flags["coronary_artery_disease"], prior_severe_hypo_12m=flags["hypoglycaemia"])
    prof["family_history_diabetes"] = any(_has_code(c.get("code"), *CONDITIONS["t2d"])
                                          for fmh in by_type.get("FamilyMemberHistory", []) for c in fmh.get("condition", []))

    on_met = on_su = on_ins = False
    insulin_units, regimen, su_freq = 0.0, "", ""
    for ms in by_type.get("MedicationStatement", []):
        if ms.get("status", "active") not in ("active", "intended", "on-hold"):
            continue
        name = _med_name(ms)
        dose, freq, unit = _daily_dose(ms)
        if "metformin" in name:
            on_met = True
            prof["metformin_dose_mg"] = dose * freq * (1000 if unit in ("g", "gram") else 1)
        elif any(w in name for w in SULFONYLUREAS):
            on_su = True
            su_freq = "BD" if freq >= 2 else "OD"
            if "glimepiride" in name:
                prof["sulfonylurea_dose_mg"] = dose * freq
            else:
                warnings.append(f"{name}: sulfonylurea dose not converted (only glimepiride doses are used)")
        elif "insulin" in name or any(w in name for w in BASAL_WORDS):
            on_ins = True
            insulin_units += dose * freq if np.isfinite(dose) else 0.0
            regimen = "premix_bid" if any(w in name for w in PREMIX_WORDS) else (regimen or "basal")
    prof.update(on_metformin=on_met, on_sulfonylurea=on_su, on_insulin=on_ins, insulin_regimen=regimen,
                sulfonylurea_frequency=su_freq)
    prof["insulin_total_daily_units"] = insulin_units if on_ins else 0.0
    if not on_met:
        prof["metformin_dose_mg"] = 0.0
    if not on_su:
        prof["sulfonylurea_dose_mg"] = 0.0
    return PatientRecord(str(pat.get("id", "patient")), prof, readings.reset_index(drop=True), warnings)


# --------------------------------------------------------------------------- writing
def _ref(kind: str, rid: str) -> dict[str, str]:
    return {"reference": f"{kind}/{rid}"}


def _iso(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")


def _obs(oid: str, pid: str, code: tuple[str, str, str], t: pd.Timestamp, **value: Any) -> dict[str, Any]:
    system, c, display = code
    return {"resourceType": "Observation", "id": oid, "status": "final", "subject": _ref("Patient", pid),
            "code": {"coding": [{"system": system, "code": c, "display": display}]}, "effectiveDateTime": _iso(t), **value}


def _qty(v: float, unit: str, code: str | None = None) -> dict[str, Any]:
    q = {"value": float(v), "unit": unit, "system": "http://unitsofmeasure.org"}
    if code:
        q["code"] = code
    return {"valueQuantity": q}


def to_bundle(patient_id: str, ehr: dict[str, Any], history: pd.DataFrame, now: pd.Timestamp) -> dict[str, Any]:
    """This project's EHR row + a sensor history frame (timestamp, cgm_mgdl, steps, heart_rate_bpm, sleep_stage)
    -> a FHIR R4 collection Bundle that parse_bundle reads back to the same model inputs."""
    now = pd.Timestamp(now)
    pid = patient_id
    lab_time = now - pd.Timedelta(days=30)
    birth = (now - pd.Timedelta(days=(float(ehr["age"]) + 0.5) * YEAR_DAYS)).date()
    entries: list[dict[str, Any]] = [{"resourceType": "Patient", "id": pid, "gender": {"M": "male", "F": "female"}.get(ehr["sex"], "unknown"),
                                      "birthDate": birth.isoformat(),
                                      "meta": {"tag": [{"code": "SYNTHETIC", "display": "Synthetic test patient, not a real person"}]}}]

    def cond(cid: str, icd: str, display: str, onset: pd.Timestamp | None = None) -> None:
        r = {"resourceType": "Condition", "id": cid, "subject": _ref("Patient", pid),
             "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
             "code": {"coding": [{"system": ICD10, "code": icd, "display": display}]}}
        if onset is not None:
            r["onsetDateTime"] = pd.Timestamp(onset).isoformat()
        entries.append(r)

    cond("t2d", "E11.9", "Type 2 diabetes mellitus without complications",
         now - pd.Timedelta(seconds=float(ehr["diabetes_duration_years"]) * YEAR_DAYS * 86400))
    for key, icd, display in (("hypertension", "I10", "Essential (primary) hypertension"),
                              ("dyslipidemia", "E78.5", "Hyperlipidaemia, unspecified"),
                              ("coronary_artery_disease", "I25.10", "Atherosclerotic heart disease")):
        if bool(ehr[key]):
            cond(key.replace("_", "-"), icd, display)
    if bool(ehr["prior_severe_hypo_12m"]):
        cond("hypo-12m", "E16.0", "Drug-induced hypoglycaemia without coma", now - pd.Timedelta(days=120))
    if bool(ehr["family_history_diabetes"]):
        entries.append({"resourceType": "FamilyMemberHistory", "id": "fh-t2d", "status": "completed", "patient": _ref("Patient", pid),
                        "relationship": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-RoleCode", "code": "PRN"}]},
                        "condition": [{"code": {"coding": [{"system": ICD10, "code": "E11", "display": "Type 2 diabetes mellitus"}]}}]})

    def med(mid: str, text: str, dose: float, unit: str, per_day: int) -> None:
        entries.append({"resourceType": "MedicationStatement", "id": mid, "status": "active", "subject": _ref("Patient", pid),
                        "medicationCodeableConcept": {"text": text},
                        "dosage": [{"timing": {"repeat": {"frequency": per_day, "period": 1, "periodUnit": "d"}},
                                    "doseAndRate": [{"doseQuantity": {"value": float(dose), "unit": unit}}]}]})

    if bool(ehr["on_metformin"]):
        med("metformin", "Metformin", float(ehr["metformin_dose_mg"]), "mg", 1)
    if bool(ehr["on_sulfonylurea"]):
        n = 2 if ehr["sulfonylurea_frequency"] == "BD" else 1
        med("glimepiride", "Glimepiride", float(ehr["sulfonylurea_dose_mg"]) / n, "mg", n)
    if bool(ehr["on_insulin"]):
        if ehr["insulin_regimen"] == "premix_bid":
            med("insulin", "Insulin premix 30/70", float(ehr["insulin_total_daily_units"]) / 2, "U", 2)
        else:
            med("insulin", "Insulin glargine (basal)", float(ehr["insulin_total_daily_units"]), "U", 1)

    for key, (code, display, unit, ucum) in {"hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%", "%"),
                                              "egfr": ("62238-1", "Glomerular filtration rate/1.73 sq M.predicted (CKD-EPI)", "mL/min/1.73m2", "mL/min/{1.73_m2}"),
                                              "ldl_mgdl": ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation", "mg/dL", "mg/dL"),
                                              "triglycerides_mgdl": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL", "mg/dL"),
                                              "bmi": ("39156-5", "Body mass index (BMI) [Ratio]", "kg/m2", "kg/m2"),
                                              "weight_kg": ("29463-7", "Body weight", "kg", "kg")}.items():
        entries.append(_obs(f"lab-{key}", pid, (LOINC, code, display), lab_time, **_qty(ehr[key], unit, ucum)))

    for i, r in enumerate(history.itertuples()):
        t = pd.Timestamp(r.timestamp)
        if np.isfinite(r.cgm_mgdl):
            entries.append(_obs(f"cgm-{i}", pid, (LOINC, "99504-3", "Glucose [Mass/volume] in Interstitial fluid"), t,
                                **_qty(r.cgm_mgdl, "mg/dL", "mg/dL")))
        if np.isfinite(r.steps):
            entries.append(_obs(f"steps-{i}", pid, (LOINC, "55423-8", "Number of steps in unspecified time Pedometer"), t,
                                **_qty(r.steps, "steps", "{steps}")))
        if np.isfinite(r.heart_rate_bpm):
            entries.append(_obs(f"hr-{i}", pid, (LOINC, "8867-4", "Heart rate"), t, **_qty(r.heart_rate_bpm, "/min", "/min")))
        if isinstance(r.sleep_stage, str) and r.sleep_stage in SLEEP_STAGES:
            entries.append(_obs(f"sleep-{i}", pid, (SLEEP_SYSTEM, "sleep-stage", "Sleep stage (wearable)"), t,
                                valueCodeableConcept={"coding": [{"system": SLEEP_SYSTEM, "code": r.sleep_stage}]}))
    return {"resourceType": "Bundle", "type": "collection", "timestamp": _iso(now),
            "entry": [{"fullUrl": f"urn:uuid:{e['resourceType'].lower()}-{e['id']}", "resource": e} for e in entries]}
