# Model card: Glycaemic Digital Twin risk model

**Research prototype. SYNTHETIC DATA. Not a medical device, not for clinical use.**

## Model details

| | |
|---|---|
| Name | Glycaemic Digital Twin: 2-hour hyper- / hypoglycaemia risk |
| Type | Two binary LightGBM classifiers (one per target), deployed in the dashboard. A PyTorch multi-task fusion net (static MLP + GRU) is a research comparator only. |
| Inputs | Last 6 h of 5-min CGM, steps, heart rate and sleep stage (37 summary features) + 24 encoded EHR features (demographics, HbA1c, eGFR, lipids, comorbidities, medication class and dose, prior severe hypo). |
| Outputs | Probability of any CGM > 180 mg/dL (hyper) and any CGM < 70 mg/dL (hypo) in the next 2 h, every 30 min. Alerts come from an **alert policy** ([alert_policy.md](alert_policy.md)): one alert per risk episode (threshold + refractory period), with operating points chosen on validation patients (hypo at a 1%, hyper at a 3% window-level false-alert rate). |
| Explanations | Per-prediction TreeSHAP drivers (LightGBM `pred_contrib`, verified against `shap.TreeExplainer`). |
| Training | `src/models/baselines.py`, seed 42, early stopping on validation patients. Hyperparameters in `config.yaml`. |
| Licence | MIT |
| Contact | Team details in the README |

## Intended use

* **Intended:** demonstrating, on synthetic data, how a digital twin can fuse an EHR profile with
  continuous wearable data to give clinicians an explained 2-hour early warning of high and low glucose
  in adults with Type 2 diabetes, with what-if exploration.
* **Users:** researchers, challenge reviewers and clinicians evaluating the *concept*.
* **Out of scope:** any real clinical decision (dosing, triage, alarms), use on real patients without
  prospective validation, Type 1 or gestational diabetes, children, hospital inpatients, and patients
  without a CGM.

## Data

* **Training and test:** 300 synthetic patients x 14 days from the project's mechanistic simulator
  ([data/README.md](../data/README.md)). Patient-level split: 194 train / 46 validation / 60 test, stratified
  by drug class. **The EHR traits are wired into the simulated glucose by design**, so the EHR stream
  carries signal by construction.
* **External check:** 14 real adults with T2D from [CGMacros](https://physionet.org/content/cgmacros/)
  (PhysioNet, CC BY-NC-SA 4.0, US cohort), CGM-only model, hyperglycaemia only
  ([external_validation.md](external_validation.md)).
* **Real linked EHR + CGM check:** 100 real adults with T2D from ShanghaiT2DM (Scientific Data 2023, CC BY
  4.0, Chinese, partly hospitalised), with the same patients' records and logged doses; models trained with
  patient-grouped cross-validation ([external_shanghai.md](external_shanghai.md)).

## Performance

Onset windows (the patient is not already past the threshold). 95% CIs resample whole patients.

| Evaluation | Target | AUROC | AUPRC (prevalence) |
|---|---|---|---|
| Synthetic test, 60 patients, deployed model | Hyper | 0.934 (0.923-0.949) | 0.739 (0.156) |
| Synthetic test, 60 patients, deployed model | Hypo | 0.987 (0.979-0.992) | 0.705 (0.022) |
| Synthetic test, nocturnal | Hypo | 0.989 (0.980-0.995) | 0.659 (0.015) |
| **Real CGM (CGMacros T2D, 14), CGM-only model, zero-shot** | Hyper | **0.698 (0.656-0.736)** | **0.463 (0.276)** |
| Real CGM, same features trained on the real cohort (leave-one-out) | Hyper | 0.771 (0.732-0.808) | 0.537 (0.276) |
| **Real linked data (ShanghaiT2DM, 100), CGM-only model, zero-shot** | Hypo | 0.870 (0.846-0.891) | 0.214 (0.027) |
| ShanghaiT2DM, CGM-only trained on the real cohort (10-fold by patient) | Hypo | 0.893 (0.871-0.914) | 0.341 (0.027) |
| ShanghaiT2DM, same, nocturnal | Hypo | 0.913 | 0.343 (0.032) |
| ShanghaiT2DM, same | Hyper | 0.871 (0.858-0.885) | 0.577 (0.147) |

* **The synthetic numbers are optimistic.** On real CGM, discrimination falls from 0.93 to 0.70 AUROC.
  Simulated lows drift in slowly (the lowest CGM in the 2 h before onset is already ~70 mg/dL), which makes
  them easy to anticipate. Hypoglycaemia could not be checked on real data: only 3 CGMacros T2D
  participants go low.
* **Calibration:** good on synthetic test data (hypo ECE 0.004), poor on real CGM (zero-shot ECE 0.14).
  Recalibrating on other real participants (leave-one-out Platt scaling, no retraining) brings it to 0.024
  with AUROC almost unchanged (0.690). Alert thresholds would still need re-tuning on real episodes.
* **EHR value: synthetic only.** +0.051 hypo AUPRC (95% CI +0.018 to +0.113) on synthetic data, partly by
  construction. **On real linked data (ShanghaiT2DM) the EHR did not improve the 2-hour forecast**: hypo
  -0.014 (-0.048 to +0.010), hyper -0.030 (-0.052 to -0.008); logged doses and a richer record did not help
  either, and a low-capacity model found no extra signal. With **no** CGM at all (a new patient) the EHR
  does carry a weak real signal about lows (hypo AUROC about 0.54 vs chance); 30 minutes of CGM already
  reaches 0.88. The deployed model still uses EHR features
  (it was trained on synthetic data), and its EHR drivers explain what it reacts to, not proven extra value.
* **Alerting, episode level** (synthetic test, [alert_policy.md](alert_policy.md)): the policy warns of **76% of
  hypo episodes at least 30 min before onset** (83% with >= 15 min; nocturnal 71%) with **0.49 hypo alerts per
  patient-day** at 66% PPV per alert and a median lead of 65 min. Hyper: 68% of episodes warned >= 30 min ahead,
  2.46 alerts per patient-day (PPV 69%), slightly above the 2-per-day cap it met on validation patients.
  The trade-off: scoring every window instead catches 98% of hypo episodes >= 30 min ahead, but at about
  3 alerting windows per patient-day.
* **Does synthetic data help real-world training?** Mostly no ([transfer_experiment.md](transfer_experiment.md)).
  On CGMacros T2D, adding synthetic data (pretraining or pooling) gives a small AUROC-only gain with 2 real
  participants, nothing from 4, and slightly hurts from 7 up. Its one consistent benefit is lower
  variance across training-set draws. Real data matters more than the simulator.
* Full results: [ablation_results.md](ablation_results.md).

## Subgroups

From [subgroups.md](subgroups.md) (synthetic test set):
* Hyper AUROC ranges from 0.916 to 0.953 across sex, age, kidney function, drug class, HbA1c and BMI groups.
* **Insulin users' lows are the hardest to predict:** hypo AUPRC 0.217, against 0.722 for sulfonylurea users.
  That group is small (7 patients, 4 with events).
* **Reduced kidney function (eGFR < 60) cannot be audited:** only 1 test patient has it, even though low
  eGFR raises hypo risk in the simulator.

## Ethical considerations and risks

* **Synthetic evidence only.** Nothing here shows clinical benefit. The EHR-to-sensor linkage and the EHR
  signal are synthetic.
* **Missed lows are the costliest error.** A false sense of safety from a low risk score could delay action.
  Any real use needs a human in the loop and must never replace standard glucose alarms.
* **Alert fatigue.** Scoring every window would cause it (at a 10% window-level false-alert rate, only ~18% of
  hypo alerts are true). The episode policy cuts hypo alerts to ~0.5 per patient-day at 66% PPV, at the cost of
  missing early warning for about 1 in 4 episodes. Where to sit on that curve is a clinical decision.
* **Automation bias.** Explanations describe what the model reacted to, not physiological causes.
* **Equity.** The synthetic cohort is loosely modelled on Indian T2D patients but is not epidemiology. Region,
  diet, socioeconomic status and device type are absent, and real performance across them is unknown.
* **Privacy.** No real patient data is used. A real deployment would process sensitive health data; see
  [deployment_path.md](deployment_path.md).

## Recommendations

Before any clinical use: validate on real paired EHR + CGM data from the target population (the transfer
experiment shows real training data matters more than the simulator), recalibrate, re-tune the alert policy
with clinicians on real episodes, complete the subgroup audit on a representative cohort,
and obtain ethics and regulatory review ([deployment_path.md](deployment_path.md)).
