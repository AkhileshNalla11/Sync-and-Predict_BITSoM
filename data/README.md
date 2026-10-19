# Data

**Everything generated in this folder is synthetic.** No real patient data is used for training. Two public, de-identified datasets (below) are used only for external checks, downloaded by the user to the git-ignored `external/` folder.
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
| Insulin / sulfonylurea (+ dose, BD vs OD) | time-varying glucose-lowering drug action -> lows after skipped/delayed meals, extra activity, extra doses, and overnight |
| Tight HbA1c on SU / insulin | closer to the 70 mg/dL threshold -> more lows (classic over-treatment risk) |
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
| `ehr.csv` | One row per patient (`patient_id`): demographics, labs, comorbidities, medications, ICD-10 codes | yes (static stream) |
| `enrollment.csv` | Study enrollment log: `patient_id` <-> `device_id` (the only link between the streams) | linkage only |
| `wearables.csv.gz` | 5-min rows keyed by `device_id`: `cgm_mgdl` (NaN = dropout), `steps`, `heart_rate_bpm` (NaN = watch off), `sleep_stage` (awake/light/deep/rem; empty outside sleep or watch off) | yes (dynamic stream) |
| `events_truth.csv` | Simulator ground truth: meals (Indian meal types, carbs), skipped/delayed meals, late dinners, doses, extra doses, activity bouts, sleep nights | **no** (audit / what-if only) |
| `sim_truth.csv.gz` | 5-min true glucose, carbs, insulin, sulfonylurea | **no** |
| `patient_params_HIDDEN.csv` | Latent simulator parameters per patient | **no** |
| `sim_summary.json` | Event rates and sanity statistics (also rendered to `docs/synthetic_data_report.md`) | - |

## Simulator summary

Bergman-minimal-model-inspired dynamics at 5-minute steps: gut absorption chains by meal type,
two-compartment PK for rapid / NPH / basal insulin and sulfonylureas, post-exercise sensitivity,
poor-sleep insulin resistance, dawn phenomenon, and an early-night window with lower insulin needs and
blunted counter-regulation. HbA1c is treated as an *on-treatment* value: drug users' untreated basal
glucose is set so their prescribed drugs bring them back to it on average. Titration is partial: it
compensates sqrt(drug sensitivity), not kidney function or age. Drug dependence is capped at 2.5x, and
doses scale with HbA1c. Lows come from timing mismatches (dose without meal, exercise, extra doses,
the early-night window) and from residual hidden over-exposure. The CGM adds a 10-min lag, autocorrelated error,
calibration bias, 40-400 mg/dL limits and dropouts.

Calibration knobs live in `config.yaml -> simulation.calibration`. Achieved rates are recorded in
`docs/synthetic_data_report.md`.

## Processed dataset (`data/processed/`, built by `python -m src.data.build_dataset`)

| File | Contents |
|---|---|
| `series.npz` | Dense raw series `(patients, 4032 steps, [cgm, steps, hr, sleep_code])`, NaN = not observed |
| `windows.csv.gz` | One row per prediction time (30-min stride): `hyper`, `hypo`, `nocturnal`, coverage, `split` |
| `static_features.csv` | Encoded static vector per patient; numeric columns standardised on **train patients only** |
| `static_encoder.json` | The scaling parameters (re-used by the twin / dashboard) |
| `splits.csv` | Patient -> drug stratum (insulin / sulfonylurea / other) -> train / val / test |
| `linkage_report.json` | Link counts, orphans, `synthetic_linkage: true` |
| `dataset_summary.json` | Window counts and positive rates per split |

**The split is by patient, never by row**, stratified by drug class (65 / 15 / 20 %). `build_dataset`
raises `PatientLeakageError` if any patient appears in two splits, and a test checks it. Windows are
sliced from `series.npz` on demand (`extract_windows` -> `encode_sequences`) instead of being stored.

Hypoglycaemia is concentrated in a minority of patients: in the test set only 17 of 60 patients ever have
a hypo window and 7 of them account for 80% of hypo windows, so hypo metrics should be read with their
patient-level bootstrap CIs.

## Optional external sources

| Source | Status | Why not used for training |
|---|---|---|
| [CGMacros v1.0.0 (PhysioNet)](https://physionet.org/content/cgmacros/) | Open access, **CC BY-NC-SA 4.0** (non-commercial, share-alike, attribution). **Used for the external sanity check only** ([docs/external_validation.md](../docs/external_validation.md)); downloaded to the git-ignored `data/external/cgmacros/`, never committed | Not used for training: 45 participants, only 14 with T2D, ~10 days; US cohort with no medication-driven lows. |
| [ShanghaiT2DM (Zhao et al., Scientific Data 2023)](https://doi.org/10.6084/m9.figshare.c.6310860) | **CC BY 4.0** (attribution). **Used for the real linked EHR + CGM check only** ([docs/external_shanghai.md](../docs/external_shanghai.md)); downloaded to the git-ignored `data/external/shanghai/`, never committed | Not used for training the deployed model: 100 Chinese T2D patients, partly hospitalised, 15-min CGM. |
| [Synthea](https://github.com/synthetichealth/synthea) | Apache 2.0 | Needs Java 17+, which is not installed here; generates US-centric EHR without CGM. Our generated EHR covers the needed fields directly. |
