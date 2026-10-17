# Data

**Everything in this folder is synthetic.** No real patient data is used anywhere in this project.
Files are not committed: regenerate them with `bash scripts/run_pipeline.sh` (deterministic given
`seed` in `config.yaml`).

## Synthetic linkage disclosure

In real life, EHR records and wearable streams for the same people are rarely available together.
Here both streams are produced by one simulator (`src/data/synthetic_gen.py`) for the same virtual
patients, and **the EHR traits are wired into the glucose dynamics on purpose** so that a fusion
model has genuine signal to find:

| EHR trait | Simulated effect |
|---|---|
| Higher HbA1c, BMI | higher insulin resistance -> larger, slower post-meal spikes; higher basal glucose |
| Insulin / sulfonylurea | time-varying glucose-lowering drug action -> lows after skipped/delayed meals, extra activity, extra doses, and overnight |
| Low eGFR | slower drug clearance -> longer, deeper lows |
| Long diabetes duration | weaker counter-regulation -> deeper lows |
| Prior severe hypo (12 m) | correlated with a hidden drug-sensitivity trait |
| Metformin only / no drugs | essentially never hypoglycaemic |

Any performance gain from adding EHR features is therefore partly **built into the data**. It shows
that the pipeline can use static context; it says nothing about how much EHR data would help with
real patients.

## Files (`data/raw/`)

| File | Contents | Model input? |
|---|---|---|
| `ehr.csv` | One row per patient: demographics, labs, comorbidities, medications, ICD-10 codes | yes (static stream) |
| `wearables.csv.gz` | 5-min rows: `cgm_mgdl` (NaN = dropout), `steps`, `heart_rate_bpm` (NaN = watch off), `sleep_stage` (awake/light/deep/rem; empty outside sleep or watch off) | yes (dynamic stream) |
| `events_truth.csv` | Simulator ground truth: meals (Indian meal types, carbs), skipped/delayed meals, late dinners, doses, extra doses, activity bouts, sleep nights | **no** (audit / what-if only) |
| `sim_truth.csv.gz` | 5-min true glucose, carbs, insulin, sulfonylurea | **no** |
| `patient_params_HIDDEN.csv` | Latent simulator parameters per patient | **no** |
| `sim_summary.json` | Event rates and sanity statistics (also rendered to `docs/synthetic_data_report.md`) | - |

## Simulator summary

Bergman-minimal-model-inspired dynamics at 5-minute steps: gut absorption chains by meal type,
two-compartment PK for rapid / NPH / basal insulin and sulfonylureas, post-exercise sensitivity,
poor-sleep insulin resistance, dawn phenomenon, and an early-night window with lower insulin needs and
blunted counter-regulation. HbA1c is treated as an *on-treatment* value: drug users' untreated basal
glucose is set so their prescribed drugs bring them back to it on average (titration). Lows come from
timing mismatches and hidden over-exposure. The CGM adds a 10-min lag, autocorrelated error,
calibration bias, 40-400 mg/dL limits and dropouts.

Calibration knobs live in `config.yaml -> simulation.calibration`. Achieved rates are recorded in
`docs/synthetic_data_report.md`.

## Optional external sources (checked, not used)

| Source | Status | Why not used (for now) |
|---|---|---|
| [CGMacros v1.0.0 (PhysioNet)](https://physionet.org/content/cgmacros/) | Open access, **CC BY-NC-SA 4.0** (non-commercial, share-alike, attribution) | 45 participants, only 14 with T2D, 10 days; US cohort with no medication-driven lows; 628 MB. Usable for a non-commercial student challenge, e.g. to sanity-check meal-response shapes, but too small for training. |
| [Synthea](https://github.com/synthetichealth/synthea) | Apache 2.0 | Needs Java 17+, which is not installed here; generates US-centric EHR without CGM. Our generated EHR covers the needed fields directly. |
