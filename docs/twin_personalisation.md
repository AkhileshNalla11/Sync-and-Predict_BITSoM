# Personalised twin: fitted parameters vs hidden truth

**SYNTHETIC DATA. Not clinically validated.** The fitted model has exactly the simulator's structure and the
meal / dose log is exact, so these numbers are a best case for a model-based twin, not an estimate of
real-world accuracy.

## What changed

The what-if engine used to replay each patient with the simulator's **hidden** parameters, schedule and
noise, which no real system could do. The default *fitted* mode now builds the twin from observable data
only (`src/twin/personalise.py`):

- **EHR** -> population-default simulator parameters (the existing `derive_patient_params` mapping, with
  the hidden latent traits and random draws set to their population typical values).
- **Patient log** -> meals (type, carbs) and medication doses; **wearable** -> steps and sleep (a short
  night raises next-day insulin resistance, as in the simulator).
- **First 7 days of CGM** -> four parameters fitted per patient by penalised least squares
  (`scipy.optimize.least_squares`, bounded trust-region Gauss-Newton, after a 64-point batched coarse
  search): untreated basal glucose `g_basal`, carbohydrate gain `meal_gain`, clearance time `tau_clear_min`,
  and drug sensitivity `drug_sens` (only for patients on insulin or a sulfonylurea). A weak prior
  (weight 5 (mg/dL)^2 per prior SD, chosen on validation patients, see below) keeps
  them near the EHR default when the data is uninformative. Everything else stays at the EHR default.
- Fit time: median 3.1 s, 90th percentile 3.9 s per patient
  (wall clock on a shared laptop CPU; candidate parameter sets are simulated side by side, at most
  9 batched simulator calls per fit; slowest 4.9 s).
  Fits are cached in `data/processed/twin_fits.csv`.
- For a what-if at time *now*, a simple state observer (gain 0.2 per 5 min) pulls the
  simulated glucose toward the CGM up to *now*; after that the model runs freely. The scenario's CGM fed
  to the risk model is the observed CGM plus the simulated scenario-minus-baseline difference.

The oracle mode is still available (dashboard toggle) for comparison.

### Choosing the prior weight (validation patients only)

Compared on the 46 validation patients with the same held-out protocol;
the test patients below played no part in any choice.

| Prior weight | Held-out CGM MAE | 2 h projection MAE | `g_basal` median rel. error | `meal_gain` median rel. error | `tau_clear_min` median rel. error | `drug_sens` median rel. error |
|---|---|---|---|---|---|---|
| 0 | 5.76 | 5.20 | 3.2% | 3.9% | 5.4% | 3.7% |
| 2 | 5.73 | 5.17 | 3.0% | 3.6% | 4.3% | 2.9% |
| 5 | 5.72 | 5.16 | 2.9% | 3.7% | 4.2% | 2.9% |
| 20 | 5.74 | 5.17 | 2.7% | 4.3% | 2.7% | 4.1% |

## Held-out validation (60 test patients, days 8-14)

All three parameter sets are driven by the same log-derived schedule without process noise, so only the
parameters differ. *Oracle* uses the hidden true parameters: an upper reference, unknowable in reality.

### 1. Open-loop simulation of the held-out week

| Parameters | CGM MAE, mg/dL (95% CI) | Bias | TIR error | TAR error | TBR error | Hyper windows: agreement / sens / kappa | Hypo windows: agreement / sens / kappa |
|---|---|---|---|---|---|---|---|
| EHR default | 12.1 (10.3-13.9) | +0.0 | 6.1 pp | 6.1 pp | 0.8 pp | 91% / 84% / 0.79 | 98% / 42% / 0.54 |
| Fitted (personalised) | 5.9 (5.6-6.1) | +0.3 | 1.6 pp | 1.4 pp | 0.3 pp | 96% / 91% / 0.90 | 98% / 52% / 0.67 |
| Oracle (hidden truth) | 7.5 (6.9-8.2) | -0.4 | 3.6 pp | 3.3 pp | 0.6 pp | 94% / 90% / 0.87 | 98% / 60% / 0.71 |

TIR / TAR / TBR error = mean absolute difference between simulated and observed time in range (70-180),
above 180 and below 70 mg/dL, in percentage points. Window events use the risk model's labels (any reading
> 180 or < 70 mg/dL in a 2 h horizon, 30 min stride; 19,671 windows). Observed event rates:
hyper 31.9%, hypo 3.5%; simulated (fitted): hyper 30.5%, hypo 1.9%.

Fitted minus EHR-default CGM MAE: -6.3 mg/dL (95% patient-bootstrap CI -8.1 to -4.9); the fit is better for 97% of patients.

### 2. Two-hour projections (what the what-if panel shows)

Every 2 h in the held-out period: observer on up to the prediction time, then a free run.
Mean absolute error against the CGM that followed.

| Method | MAE at 30 min | at 60 min | at 120 min | mean over 5-120 min |
|---|---|---|---|---|
| CGM persistence | 10.7 | 19.1 | 29.7 | 18.5 |
| EHR default | 8.0 | 9.6 | 10.9 | 9.2 |
| Fitted (personalised) | 4.9 | 5.4 | 5.8 | 5.2 |
| Oracle (hidden truth) | 5.4 | 6.2 | 6.9 | 5.9 |

Persistence holds the last CGM reading flat. The mechanistic projections know the logged meals and doses
in the next 2 h (the what-if panel's scenarios also edit future events), which persistence cannot use.

### 3. Does the fit recover the hidden parameters?

| Parameter | Patients | Median relative error: EHR default | fitted | Spearman with truth: EHR default | fitted |
|---|---|---|---|---|---|
| `g_basal` | 60 | 2.7% | 2.4% | 0.99 | 0.98 |
| `meal_gain` | 60 | 7.7% | 3.6% | 0.91 | 0.97 |
| `tau_clear_min` | 60 | 2.3% | 3.3% | 0.91 | 0.75 |
| `drug_sens` | 25 | 16.8% | 4.4% | 0.84 | 0.98 |

`drug_sens` is scored on patients taking insulin or a sulfonylurea only (it has no effect otherwise).

## Caveats

- **Best case by construction.** The fitted model is the simulator itself with four free parameters;
  real physiology would add model mismatch. The log here is exact; real carb counts are often off by
  20-50 %, and doses and meal times get forgotten.
- **Hidden drivers remain unmodelled.** The twin cannot see the simulator's process noise, the true
  sleep-driven insulin-resistance swings on nights the watch was not worn, or sensor calibration error;
  this is why even oracle parameters leave a sizeable open-loop error.
- **Fitted can beat oracle on CGM error, and that is not a contradiction.** The hidden parameters are the
  best *physiological* description, but a deterministic replay of them still misses the per-sensor
  calibration bias and the average effect of hidden drivers. Fitted parameters absorb those, so part of
  the gain is calibration of the sensor and of the model, not better physiology. Table 3 is the
  physiological check: there the fit improves on the EHR default mainly for carbohydrate gain and drug
  sensitivity, and does not beat it for every parameter.
- **The EHR-to-parameter mapping is the simulator's own,** so the EHR default is unrealistically good; in
  a real population far more between-patient variation would be left for the fit to explain.
- **Identifiability.** `g_basal` and `tau_clear_min` (and to a lesser degree `meal_gain`) partly trade
  off against each other: together they set the average level and the size of meal excursions, and the
  prior holds them near the EHR default along that ridge. Fitted `tau_clear_min` is in fact slightly
  further from the truth than the EHR default (table 3). Individual values should not be read as
  physiological measurements; drug sensitivity, which dose timing pins down, is the best identified.
- **Not a clinical tool.** Nothing here is clinically validated; the what-if projections are illustrations
  of the synthetic model.

Reproduce: `python -m src.eval.twin_validation --tune 0,2,5,20` (validation patients, writes
`docs/twin_personalisation_tuning.csv`), then `python -m src.eval.twin_validation` (test patients, writes
this file and `docs/twin_personalisation.csv`).
