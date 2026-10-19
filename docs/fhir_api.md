# FHIR risk API

**Research prototype. Not a medical device, not for clinical use.** The example data is synthetic.

The dashboard reads precomputed files. A hospital system, a CGM app or an ABDM-connected health locker would
instead send a patient's record and recent sensor readings in **FHIR R4**, the format India's ABDM health
records use, and expect the result back in FHIR. This API does that: one FHIR `Bundle` in, the 2-hour risks
out as JSON and as FHIR `RiskAssessment` resources.

```bash
uvicorn src.api.app:app --port 8000        # or scripts\api.cmd / bash scripts/api.sh
# interactive docs: http://localhost:8000/docs
curl -X POST "http://localhost:8000/v1/risk?model=real" -H "Content-Type: application/json" \
     --data @examples/fhir/P0014_bundle.json
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness, and which models are loaded |
| `GET /v1/models` | Each model's inputs, training data and alert thresholds |
| `POST /v1/risk?model=real\|synthetic&now=...` | FHIR Bundle in, JSON risks with the top 3 drivers per event out |
| `POST /v1/risk/fhir?model=...` | The same, returned as a FHIR Bundle of two `RiskAssessment` resources |

* **`model=real`** (default): the CGM-only model trained and alert-tested on 100 real patients
  ([real_model.md](real_model.md)). It needs only CGM.
* **`model=synthetic`**: the CGM + wearables + EHR model trained on this project's synthetic patients. Missing
  EHR fields are scored as unknown and listed in `warnings`.
* **`now`**: the prediction time; by default, the latest CGM reading.

## What the API reads from the Bundle

| Resource | Codes | Becomes |
|---|---|---|
| `Patient` | `gender`, `birthDate` | sex, age |
| `Observation` CGM | LOINC **99504-3** (glucose in interstitial fluid), mg/dL or mmol/L | the 6 h glucose window |
| `Observation` labs | HbA1c 4548-4 / 17856-6 (% or mmol/mol); eGFR 62238-1 / 33914-3 / 98979-8 / 48642-3 / 48643-1; LDL 13457-7 / 18262-6 / 2089-1 and triglycerides 2571-8 (mg/dL or mmol/L); BMI 39156-5; weight 29463-7 / 3141-9 | the latest value of each |
| `Observation` wearable | steps 55423-8; heart rate 8867-4; sleep stage (local code system, see below) | steps, heart rate, sleep |
| `Condition` (ICD-10 or SNOMED CT) | type 2 diabetes E11 / 44054006 (onset gives the duration); hypertension I10-I15 / 38341003; dyslipidaemia E78 / 55822004; coronary disease I20-I25 / 53741008; hypoglycaemia E16.0-E16.2, E11.64 / 302866003 within 12 months | the EHR flags; resolved or inactive conditions are ignored |
| `FamilyMemberHistory` | type 2 diabetes in a relative | family history |
| `MedicationStatement` | metformin; sulfonylureas; insulin (premixed or basal), with dose and frequency | drug classes and daily doses |

**Limits of the mapping:**
* Sulfonylurea doses are kept only for glimepiride, because the model's dose feature is in glimepiride
  units. Other sulfonylureas count as "on a sulfonylurea" with an unknown dose, and the API says so in
  `warnings`.
* There is no standard LOINC code for a per-epoch sleep stage. Sleep uses a local code system
  (`.../fhir/sleep-stage`: awake, light, deep, rem).
* Only the real-patient-trained model is meant for real data. It uses CGM and time of day only.

## How the window is built

Exactly as in training. Readings from the last 6 hours before `now` are snapped to a 5-minute grid. Sparser
sensors (e.g. a 15-minute CGM) are linearly interpolated between readings at most 30 minutes apart, which is
the same preprocessing the real-patient model was trained with on ShanghaiT2DM. If less than 50% of the window
has CGM (the training minimum), the API returns **422** rather than guess. A Bundle without exactly one
`Patient` returns **400**.

## What comes back

```json
{
  "patient_id": "P0014",
  "assessed_at": "2026-01-18T09:55:00",
  "horizon_hours": 2,
  "model": {"name": "lgbm_cgm_shanghai", "inputs": "CGM + time of day", "trained_on": "ShanghaiT2DM"},
  "risks": {
    "hypo": {
      "probability": 0.204, "alert_threshold": 0.293, "above_alert_threshold": false,
      "drivers": [
        {"factor": "Current glucose", "value": "70 mg/dL", "effect": "raises risk", "log_odds": 2.2452},
        {"factor": "Lowest glucose, 6 h", "value": "64 mg/dL", "effect": "raises risk", "log_odds": 0.7011},
        {"factor": "Change over last 15 min", "value": "+6 mg/dL", "effect": "lowers risk", "log_odds": -0.6885}
      ]
    },
    "hyper": {"probability": 0.222, "alert_threshold": 0.520, "above_alert_threshold": false, "drivers": ["..."]}
  },
  "data": {"cgm_readings": 72, "native_interval_min": 5.0, "cgm_coverage": 1.0, "history_hours": 6.0},
  "warnings": [],
  "disclaimer": "Research prototype. Not a medical device and not for clinical use. ..."
}
```

`/v1/risk/fhir` returns the same as two `RiskAssessment` resources, ready to be written back to a FHIR server:
* the outcome is coded in SNOMED CT (302866003 hypoglycaemia, 80394007 hyperglycaemia);
* `probabilityDecimal` holds the risk;
* `qualitativeRisk` is `high` when the risk is at or above the alert threshold, `low` otherwise;
* `whenPeriod` covers the next 2 hours;
* the drivers go in `rationale`;
* the disclaimer goes in `note`.

## Design choices

* **Stateless:** nothing is stored. The episode alert policy (one alert per episode, 90-minute refractory)
  needs the history of previous alerts, so a caller applies it across successive calls
  ([src/twin/alerts.py](../src/twin/alerts.py)). The API reports whether each risk is above the threshold.
* **Tested for exactness:** a synthetic patient exported to FHIR (`src/interop/fhir.py: to_bundle`) and scored
  through the API gets the same risk as the dashboard, to 1e-9, for both models
  ([tests/test_interop.py](../tests/test_interop.py)). CI runs this on every push.
* **Where it would sit in ABDM:** behind a Health Information User consent flow. The patient consents through
  ABHA, the health locker releases the Bundle, and the RiskAssessment returns to the clinician's system. That
  wiring, authentication, audit logs and FHIR profile validation against the NRCeS / ABDM profiles are not
  built here ([deployment_path.md](deployment_path.md)).
