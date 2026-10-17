# External sanity check on real CGM data (CGMacros)

**A sanity check, not a clinical validation.** Everything else in this project is trained and evaluated on
synthetic data. Here the synthetic-trained CGM-only model is run, unchanged, on real CGM traces from a
small public US cohort, to see whether it learned anything that transfers. Nothing here supports clinical
use. The synthetic pipeline and its reported results are unchanged.

Numbers: [external_validation.csv](external_validation.csv) (all models, cohorts and subsets),
[external_validation_paired.csv](external_validation_paired.csv) (paired differences) and
[external_validation_cohort.csv](external_validation_cohort.csv) (aggregated cohort description).

## Data

[CGMacros v1.0.0](https://physionet.org/content/cgmacros/) (PhysioNet, open access, **CC BY-NC-SA 4.0**):
45 US adults wore an Abbott FreeStyle Libre Pro and a Dexcom G6 Pro at the same time for about 10 days,
plus a Fitbit, with logged meals. The data is used here only for this non-commercial check. It is **not**
redistributed: it lives in the git-ignored `data/external/cgmacros/`, and tests use a small made-up fixture
in the same file format. The files in this folder contain only aggregate metrics.

Cohort, split by lab HbA1c with the dataset's own cut-offs (healthy < 5.7 %, prediabetes 5.7-6.4 %,
T2D > 6.4 %):

| | T2D (headline) | Prediabetes | Healthy |
|---|---|---|---|
| Participants | 14 | 16 | 15 |
| HbA1c, median (range) | 7.1 % (6.5-8.5) | 5.95 % (5.7-6.4) | 5.3 % (4.6-5.6) |
| Monitored days (Dexcom grid) | 139 | 157 | 157 |
| CGM coverage | 99.0 % | 95.3 % | 95.7 % |
| Prediction windows (30-min stride) | 6,438 | 6,963 | 6,811 |
| Hyper onset windows (positive / prevalence) | 1,207 / 0.276 | 944 / 0.150 | 489 / 0.074 |
| Participants with any hyper window | 14 | 16 | 13 |
| Hypo windows (all) | **31** | 56 | 194 |
| Participants with any hypo window | **3** | 5 | 11 |

The cohort is 34 Hispanic/Latino, 7 White and 4 African American participants. T2D participants have a
median age of 52.5 and a median BMI of 30.6. `bio.csv` has no medication data.

## Method

* **Loader** ([`src/data/cgmacros.py`](../src/data/cgmacros.py)):
  * **One sensor for everyone: Dexcom G6.** It samples every 5 min, like our synthetic CGM. The Libre
    Pro samples every 15 min. The published files hold both sensors already linearly interpolated onto a
    1-min grid.
  * **Resampling:** the value at each 5-min clock mark, on a regular grid from first to last reading.
    Missing rows become NaN, as in our synthetic series.
  * **Gaps:**
    * Interpolation hides dropouts, so exactly straight stretches of ≥ 30 min count as bridged gaps
      and are set to NaN: 5,224 minutes in total, under 1 % of the data.
    * Two participants also have long missing stretches, 9.4 days (healthy) and 40 h (prediabetes).
      Windows there are dropped by the usual coverage rules.
    * No reading was outside 40-400 mg/dL.
* **Windows and labels:** the unchanged `label_windows` from
  [`wearable_features.py`](../src/data/wearable_features.py). Each window has 6 h of history and a
  2 h horizon, with a 30-min stride.
  * **Labels:** hyper = any CGM > 180 mg/dL in the horizon, hypo = any < 70.
  * **Coverage:** windows need ≥ 50 % observed CGM in both history and horizon.
  * ***Onset*** (headline) excludes windows that are already past the threshold.
* **Features:** the unchanged `handcrafted_features`, CGM and time-of-day groups only (24 features).
  Our EHR vector has no CGMacros equivalent. The Fitbit stream was not mapped onto our steps/HR/sleep
  features, which added nothing on synthetic data anyway.
* **Models:**
  * **Persistence:** extrapolation of the 30-min trend; no training.
  * **LightGBM CGM-only, zero-shot:** `lgbm_cgm_{hyper,hypo}.txt`, trained only on synthetic
    patients and applied as is. No recalibration, no threshold tuning.
  * **LightGBM CGM-only, leave-one-participant-out (LOPO), T2D only:** a reference for what the same
    features reach when trained on real data. Same hyperparameters. Each held-out participant is scored by
    a model trained on the other 13: 10 for training, 3 for early stopping. Out-of-fold scores are pooled.
    Hyper only: 31 hypo windows are too few to train on.
* **Metrics:** the unchanged [`src/eval/metrics.py`](../src/eval/metrics.py): AUPRC (chance = prevalence),
  AUROC, and sensitivity / PPV at a 10 % false-alert rate.
  * The threshold is set on these same windows, so the operating-point numbers are optimistic.
  * 95 % CIs come from a patient-level bootstrap with 200 resamples. Model comparisons are paired on the
    same resamples.

## Results: T2D, hyperglycaemia (> 180 mg/dL in the next 2 h)

Onset windows, 14 participants, 1,207 positive of 4,377 (prevalence 0.276). For comparison, the synthetic
test row: same model, same features, 60 synthetic test patients.

| Model | AUPRC (95 % CI) | AUROC (95 % CI) | Sens @ 10 % FAR | PPV @ 10 % FAR |
|---|---|---|---|---|
| *chance (prevalence)* | 0.276 | 0.5 | 0.10 | |
| Persistence | 0.438 (0.370-0.522) | 0.659 (0.618-0.691) | 0.240 | 0.479 |
| **LightGBM CGM-only, synthetic-trained, zero-shot** | **0.463** (0.388-0.547) | **0.698** (0.656-0.736) | 0.266 | 0.504 |
| LightGBM CGM-only, LOPO on CGMacros T2D | 0.537 (0.474-0.614) | 0.771 (0.732-0.808) | 0.397 | 0.603 |
| *Synthetic test set, LightGBM CGM-only (prevalence 0.156)* | *0.724 (0.679-0.757)* | *0.930 (0.919-0.943)* | *0.757* | |

Paired differences on onset windows (A - B, same patient resamples):

| Comparison | AUPRC | AUROC |
|---|---|---|
| Zero-shot vs persistence | +0.025 (-0.004 to +0.055) | **+0.039** (+0.014 to +0.065) |
| LOPO vs persistence | **+0.100** (+0.059 to +0.149) | **+0.112** (+0.086 to +0.145) |
| LOPO vs zero-shot | **+0.074** (+0.027 to +0.124) | **+0.073** (+0.049 to +0.099) |

Bold = 95 % CI excludes zero. On *all* windows, which include those already above 180, every model scores
much higher (zero-shot AUPRC 0.909 at prevalence 0.501), because the current reading nearly decides the label.

Calibration, onset windows:

| Model | Mean predicted | Observed | ECE |
|---|---|---|---|
| Zero-shot | 0.225 | 0.276 | 0.136 |
| LOPO | 0.291 | 0.276 | 0.023 |

**Reading this honestly**
* **Something transfers, but not much.**
  * The synthetic-trained model ranks real windows better than chance: AUROC 0.70, AUPRC 1.7x
    prevalence. It beats persistence on AUROC.
  * Its AUPRC gain over persistence (+0.025) is small, and the CI includes zero.
* **The synthetic numbers overstate real-world performance by a wide margin.**
  * Onset AUROC falls from 0.93 on synthetic patients to 0.70 on real ones.
  * Sensitivity at a 10 % false-alert rate falls from 0.76 to 0.27.
  * This matches the warning in the README that simulated glucose is smoother and more regular than real glucose.
* **Real data helps.** The same features and hyperparameters, trained on just 10-13 real T2D
  participants, beat the zero-shot model by +0.07 AUPRC. That still falls well short of the synthetic numbers.
  The simulator's meal responses and daily rhythm are not a good enough stand-in for real ones.
* **Zero-shot probabilities are miscalibrated** on this cohort: they underpredict onset risk (ECE 0.14).
  The deployed alert thresholds would need recalibrating before any use on real CGM.

### Prediabetes and healthy (zero-shot, hyper onset)

| Cohort | Prevalence | Persistence AUPRC | Zero-shot AUPRC | Zero-shot AUROC |
|---|---|---|---|---|
| Prediabetes (16) | 0.150 | 0.234 (0.174-0.303) | 0.325 (0.235-0.392) | 0.712 (0.682-0.744) |
| Healthy (15) | 0.074 | 0.171 (0.125-0.233) | 0.232 (0.160-0.312) | 0.747 (0.680-0.795) |

These cohorts are outside the model's intended T2D population, and the event is rarer. They are shown
only to make clear that the T2D picture is not specific to that subgroup.

## Hypoglycaemia: too few events to evaluate

| Cohort | Hypo windows | Onset positives | Participants with any | Zero-shot onset AUPRC (95 % CI) |
|---|---|---|---|---|
| T2D | 31 | 29 | 3 | 0.265 (0.015-0.501) |
| Prediabetes | 56 | 44 | 5 | 0.252 (0.130-0.359) |
| Healthy | 194 | 167 | 11 | 0.113 (0.048-0.210) |

**Do not read these as hypo performance.**
* In T2D, all positives come from 3 participants, so the "CI" spans almost the whole range.
* The nocturnal-hypo rows in the CSV have only 16 positives. Their bootstrap intervals are conditional
  on resamples that happen to include those participants, so they are not meaningful.
* This cohort has none of the insulin- or sulfonylurea-driven lows that our synthetic hypo task models.
  Healthy participants show *more* sub-70 readings than T2D participants. These are more likely
  normal physiology, sensor error at the low end or compression artefacts than clinically relevant
  hypoglycaemia.
* The synthetic hypo results therefore remain untested on real data.

## Sensor sensitivity (Libre instead of Dexcom)

Re-running with `--sensor libre` changes the *labels* substantially, because in this dataset the two
sensors disagree: the Libre reads lower.

| | Dexcom | Libre |
|---|---|---|
| T2D hyper onset prevalence | 0.276 | 0.165 |
| Hypo windows: T2D | 31 | 226 |
| Hypo windows: healthy | 194 | 2,318 |

The model comparison keeps its direction on T2D hyper onset:

| Model | AUPRC with Libre |
|---|---|
| Persistence | 0.380 |
| Zero-shot | 0.399 |
| LOPO | 0.453 |

The Libre series is a 15-min signal interpolated to 1 min, so it is smoother. The Libre results are not
saved in `docs/`; the command below reproduces them.

## Caveats

* **Small n.**
  * There are 14 T2D participants, about 10 days each.
  * Windows are highly correlated within a participant, so the effective sample is closer to 14 than to 6,438.
  * The CIs reflect this. Differences of a few AUPRC points are not resolvable.
* **Different population.**
  * The cohort is US-based and mostly Hispanic/Latino, eating US meals.
  * Our simulator targets Indian T2D patients and Indian meal patterns.
  * HbA1c is mild (median 7.1 %), and there is no medication information.
* **No medication-driven lows,** so the hypoglycaemia task, the one with the most clinical weight in
  this project, cannot be checked here.
* **Different sensor.**
  * The Dexcom G6 is worn on the abdomen. The published values are pre-interpolated to 1 min, so
    dropouts can only be inferred.
  * Labels depend on the sensor choice; see above.
* **Operating points are optimistic.** The 10 % false-alert thresholds were set on the evaluation windows themselves.
* **LOPO uses one fixed hyperparameter set** taken from the synthetic pipeline, with early stopping on
  3 participants. It is a reference point, not a tuned model.
* **Not a clinical validation** of any kind.

## Reproduce

Download `CGMacros_dateshifted365.zip` (627 MB) from <https://physionet.org/content/cgmacros/>, or from
PhysioNet's open-data mirror `s3://physionet-open/cgmacros/1.0.0/`, which is much faster here. Check it
against the published `SHA256SUMS.txt`, then extract only the CSVs:

```bash
mkdir -p data/external/cgmacros && cd data/external/cgmacros
unzip -q CGMacros_dateshifted365.zip 'CGMacros/*.csv' 'CGMacros/CGMacros-*/CGMacros-*.csv' -d .
cd ../../..
python -m src.eval.external_cgmacros --data-dir data/external/cgmacros   # writes docs/external_validation*.csv
python -m src.eval.external_cgmacros --sensor libre --no-lopo --out-dir /tmp   # sensor sensitivity
```

The trained synthetic models must exist in `data/processed/models/` (`bash scripts/run_pipeline.sh`).
Most of the run time is the 14 LOPO folds.
