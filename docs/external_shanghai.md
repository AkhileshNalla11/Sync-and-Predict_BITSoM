# Real linked EHR + CGM check (ShanghaiT2DM)

**A research check on real data, not a clinical validation.** Everywhere else in this project the EHR and
the CGM are linked synthetically. ShanghaiT2DM is the first dataset here where both come from **the same
real patients**, so it tests the project's central claim: does adding the EHR to the CGM improve a 2-hour
forecast of high and low glucose?

**Short answer: no, not once there is any CGM.** On 100 real patients, models trained on the CGM alone predict
real highs and lows well (hypo onset AUROC 0.89, nocturnal 0.91). Adding the EHR, the logged doses or a
richer clinical record does not improve them, and sometimes makes them slightly worse. The EHR gain on
synthetic data (+0.051 hypo AUPRC) came largely from the simulator, which wires EHR traits into glucose
by design. Only for a new patient with **no** CGM yet does the EHR carry a small real signal about lows
(hypo AUROC about 0.54 instead of chance): useful for deciding who to watch, too weak to raise alerts on.

Numbers: [external_shanghai.csv](external_shanghai.csv) (all models and subsets),
[external_shanghai_paired.csv](external_shanghai_paired.csv) (paired differences),
[external_shanghai_cohort.csv](external_shanghai_cohort.csv) (aggregate cohort description) and
[external_shanghai_importance.csv](external_shanghai_importance.csv) (gain share by data stream).
Reproduce: `python -m src.eval.external_shanghai --data-dir data/external/shanghai`.

## Data

ShanghaiT2DM (Zhao Q. et al., *Chinese diabetes datasets for data-driven machine learning*, Scientific
Data 10:35, 2023; [figshare](https://doi.org/10.6084/m9.figshare.c.6310860), **CC BY 4.0**). The data is
**not** redistributed: it lives in the git-ignored `data/external/shanghai/`. Tests use a small made-up
fixture in the same file format, and the files in this folder hold aggregate metrics only.

| | |
|---|---|
| Patients / recording periods | 100 / 109 (some patients were recorded twice) |
| Monitored days | 1,169 (median 12.7 days per period) |
| CGM | every 15 min, put on our 5-min grid by linear interpolation (gaps > 30 min stay missing); 99.99 % coverage |
| Median age / HbA1c / eGFR | 63 years / 8.3 % / 107 |
| On insulin / sulfonylurea / metformin | 51 % / 14 % / 43 % of periods |
| Prediction windows (30-min stride) | 54,441 |
| Hyper onset windows (positive / prevalence) | 6,523 / 0.147; 97 patients have one |
| **Hypo onset windows** | **1,461 / 0.027; 65 patients have one** (CGMacros T2D had 3) |
| Nocturnal hypo onset windows | 573 |

* **Record per period:** sex, age, BMI, weight, diabetes duration, HbA1c, eGFR, lipids, complications,
  comorbidities and drug names, plus C-peptide, insulin, glycated albumin and other labs.
* **Time-stamped log:** meals, subcutaneous insulin doses, oral agents, insulin pump (CSII) and IV insulin.
  35 periods include pump insulin and 6 include IV insulin, so part of the cohort was probably treated in
  hospital, which this project's model card lists as out of scope.
* **Population:** a Chinese hospital cohort, not Indian; diet, devices and care differ.

## Method

* **Loader** ([`src/data/shanghai.py`](../src/data/shanghai.py)):
  * **CGM:** readings outside 40-400 mg/dL are clipped to that range, as our synthetic sensor does (359
    readings); dropping them would remove the deepest lows. Linear interpolation cannot create a new high or
    low, so labels match those on the raw readings.
  * **EHR mapping onto our 24 features:**
    * HbA1c converted from mmol/mol to % (÷ 10.929 + 2.15); LDL × 38.67 and triglycerides × 88.57 to mg/dL.
    * Hypertension, dyslipidaemia (hyperlipidaemia) and coronary disease come from the comorbidity and
      complication fields.
    * Drug classes come from the drug names: metformin, sulfonylureas, insulin, premixed or basal insulin.
    * Doses, family history and twice-daily sulfonylurea have no equivalent and are left missing.
  * **The summary's "Hypoglycemia (yes/no)" field is not used.** All 10 "yes" periods had CGM lows, but so
    did 59 "no" periods. It is unclear whether it describes history or this recording, so it could leak the
    outcome.
  * **Logged doses** become time-varying features: insulin units in the last 2 h and 6 h, a secretagogue
    (sulfonylurea or glinide) or any oral agent in the last 6 h, pump or IV insulin in the last 6 h, and a
    meal logged in the last 2 h. These only use log entries up to the prediction time.
* **Windows, labels and features:** the unchanged `label_windows` and `handcrafted_features` (6 h history,
  2 h horizon, 30-min stride; "onset" excludes windows already past the threshold).
* **Models:**
  * *Persistence* and the *synthetic-trained CGM-only LightGBM applied unchanged* (zero-shot).
  * Four LightGBMs trained on ShanghaiT2DM with the project's hyperparameters: **CGM only**, **+ our 24 EHR
    features**, **+ logged doses**, and **+ the full clinical record** (11 extra labs, 4 complications,
    6 more drug classes).
  * **Patient-grouped 10-fold cross-validation:** every patient is scored by models that never saw any of
    their periods; early stopping uses 15 % of the training patients. All models share the folds.
  * **Low-capacity check:** trees can memorise 100 patients from their static features, which could hide
    real signal. So we also fit a logistic regression on the CGM model's out-of-fold logit, with and
    without 8 core EHR fields (drug classes, HbA1c, eGFR, age, BMI, duration).
* **Metrics:** the unchanged `src/eval/metrics.py`. 95 % CIs come from a patient bootstrap (200
  resamples). Comparisons are paired on the same resamples.

## Results

Onset windows. Hypo prevalence 0.027, hyper 0.147.

| Model | Hypo AUPRC | Hypo AUROC | Nocturnal hypo AUROC | Hyper AUPRC | Hyper AUROC |
|---|---|---|---|---|---|
| *chance* | *0.027* | *0.5* | *0.5* | *0.147* | *0.5* |
| Persistence | 0.146 (0.113-0.182) | 0.832 | 0.871 | 0.373 (0.346-0.407) | 0.754 |
| Synthetic-trained, zero-shot | 0.214 (0.176-0.255) | 0.870 | 0.898 | 0.383 (0.352-0.422) | 0.732 |
| **Trained on real data: CGM only** | **0.341** (0.275-0.401) | **0.893** | **0.913** | **0.577** (0.542-0.615) | **0.871** |
| + our 24 EHR features | 0.327 (0.274-0.376) | 0.888 | 0.921 | 0.547 (0.510-0.591) | 0.858 |
| + EHR + logged doses | 0.321 (0.273-0.370) | 0.893 | 0.917 | 0.547 (0.516-0.590) | 0.858 |
| + full clinical record | 0.300 (0.253-0.345) | 0.874 | 0.910 | 0.556 (0.515-0.600) | 0.860 |

Paired differences (A - B), onset windows:

| Comparison | Hypo AUPRC | Hyper AUPRC | Nocturnal hypo AUPRC |
|---|---|---|---|
| + EHR vs CGM only | -0.014 (-0.048 to +0.010) | **-0.030** (-0.052 to -0.008) | +0.017 (-0.010 to +0.048) |
| + EHR + doses vs + EHR | -0.006 (-0.015 to +0.006) | +0.000 (-0.008 to +0.010) | -0.012 (-0.026 to +0.006) |
| + full record vs CGM only | **-0.041** (-0.081 to -0.009) | -0.021 (-0.044 to +0.001) | -0.017 (-0.043 to +0.006) |
| Low-capacity: CGM logit + 8 EHR fields vs CGM logit | -0.010 (-0.024 to +0.004) | -0.002 (-0.013 to +0.010) | -0.001 (-0.021 to +0.022) |
| Zero-shot vs persistence | **+0.069** (+0.035 to +0.106) | +0.010 (-0.014 to +0.033) | **+0.103** (+0.056 to +0.158) |
| CGM only (real) vs zero-shot | **+0.127** (+0.089 to +0.163) | **+0.193** (+0.170 to +0.218) | **+0.078** (+0.036 to +0.119) |

Bold = 95 % CI excludes zero. Calibration of the real-trained models is good (onset ECE 0.004-0.008);
zero-shot hyper is not (0.084).

Insulin users, hypo onset (51 patients on insulin, 457 positive windows; 50 not on insulin, 1,004):

| Model | AUPRC, on insulin | AUPRC, not on insulin |
|---|---|---|
| CGM only | 0.270 | 0.375 |
| + EHR | 0.278 | 0.351 |
| + EHR + logged doses | 0.276 | 0.343 |

## Reading this honestly

* **The EHR does not improve a 2-hour CGM forecast here.**
  * No EHR variant beats CGM-only. The tree models get slightly worse; the low-capacity model is flat.
  * The EHR does get used: 9 % of the model's gain, led by HbA1c, age, BMI and triglycerides. But it adds nothing
    the last 6 hours of CGM do not already show.
  * Logged insulin and oral-drug doses add nothing either, even for insulin users. That matches the
    synthetic finding that insulin users' lows are the hardest to predict.
* **This replaces a synthetic result rather than confirming it.** On synthetic data the EHR raised hypo
  AUPRC by +0.051, but the simulator builds EHR traits into glucose by design. On real linked data that gain
  does not appear. The project's headline claim about the EHR has been changed accordingly.
* **Where the EHR may still matter:** with no CGM at all it carries a small real signal about lows (next
  section), and, not tested here, over longer horizons than 2 h, in much larger cohorts where patient-level
  effects can be learned without memorising, and for explaining a risk to a clinician.
* **The CGM-based approach works on real data**, including for lows: hypo onset AUROC 0.89 and nocturnal
  0.91, at about 12x the prevalence in AUPRC. These are real lows in a cohort where half the patients use
  insulin.
* **The synthetic-trained model transfers better for lows than for highs.** Zero-shot, it beats persistence
  on hypo (+0.07 AUPRC, nocturnal +0.10) but not on hyper. A model trained on real data is far better
  for both (+0.13 and +0.19 AUPRC).
* **Caveats:**
  * 100 patients, one centre, Chinese and partly hospitalised: not the Indian outpatient population this
    project targets.
  * The 15-min CGM was interpolated to 5 min, which smooths short dips.
  * Cross-validation was run once, on one fold split.

## Cold start: a new patient with little or no CGM

Does the EHR help when there is not yet 6 hours of CGM? For each history length, every window keeps only
its last 0, 0.5, 1, 2 or 6 hours of CGM and the CGM features are recomputed from that. At 0 hours only the
time of day is left. The windows, labels, folds and paired bootstrap are unchanged, so rows are comparable
across history lengths. Reproduce: `python -m src.eval.external_shanghai_coldstart`; numbers in
[external_shanghai_coldstart.csv](external_shanghai_coldstart.csv).

Hypo onset (prevalence 0.027). "Low-capacity" is the logistic regression on the CGM model's logit, without
and with 8 core EHR fields:

| CGM history | AUPRC, CGM only | AUPRC, + EHR (LightGBM) | AUPRC, + EHR (low-capacity) | AUROC, CGM only -> + EHR (low-capacity) |
|---|---|---|---|---|
| **0 h** (time of day only) | 0.026 | **0.038** | **0.032** | 0.41 -> **0.54** |
| 30 min | 0.309 | 0.210 | 0.296 | 0.88 -> 0.87 |
| 1 h | 0.311 | 0.213 | 0.297 | 0.88 -> 0.88 |
| 2 h | 0.324 | 0.254 | 0.311 | 0.90 -> 0.89 |
| 6 h | 0.341 | 0.327 | 0.327 | 0.89 -> 0.89 |

Paired EHR effect with **no CGM** (A - B, AUPRC): hypo onset +0.012 (0.000 to +0.029) in LightGBM and
+0.011 (+0.002 to +0.025) in the low-capacity model; nocturnal hypo +0.020 (+0.001 to +0.082) and +0.014
(+0.002 to +0.039). Hyper: -0.002 (-0.044 to +0.044) and +0.026 (-0.010 to +0.065).

* **The EHR carries a small, real signal about lows, visible only when there is no CGM.** With the sensor
  not yet running, the EHR lifts hypo AUROC from chance to about 0.54 (nocturnal 0.58), and the CIs
  exclude zero. That is too weak to raise 2-hour alerts on. It is risk stratification ("this patient is
  somewhat more likely to go low"), not forecasting.
* **30 minutes of CGM is already almost as good as 6 hours** (hypo AUROC 0.88 vs 0.89). Once any CGM
  exists, the low-capacity model shows no EHR gain at any history length.
* **Tree models with EHR features get clearly worse with short history** (hypo AUPRC -0.10 at 30 min, CI
  excludes zero). With little CGM to go on, LightGBM leans on the static features and learns to recognise
  the 100 training patients instead of general patterns. More patients, or a model that treats the EHR as
  a weak prior, would be needed to use it safely.
* **Hyperglycaemia without CGM is driven by time of day** (AUROC 0.70, meals), and the EHR adds nothing
  significant.

## How it changes the project

* The model card, README and deployment path now say that the EHR gain is **synthetic only**, and that on
  real linked data the EHR did not improve the 2-hour forecast once any CGM was available.
* The EHR's realistic roles are now stated as: weak risk stratification before a sensor is placed, a prior
  for personalising the twin, and context for explaining a risk.
* The dashboard still shows EHR drivers, because they explain what the deployed (synthetic-trained) model
  reacts to. That is not evidence that the EHR improves prediction.
* Validating on real Indian linked records (deployment path step 1a) matters even more now. It should test
  the EHR's value directly, as here, rather than assume it.
