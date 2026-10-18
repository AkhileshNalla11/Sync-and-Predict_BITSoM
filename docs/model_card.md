# Model card: Glycaemic Digital Twin risk model

**Research prototype. SYNTHETIC DATA. Not a medical device, not for clinical use.**

## Model details

| | |
|---|---|
| Name | Glycaemic Digital Twin: 2-hour hyper- / hypoglycaemia risk |
| Type | Two binary LightGBM classifiers (one per target), deployed in the dashboard. A PyTorch multi-task fusion net (static MLP + GRU) is a research comparator only. |
| Inputs | Last 6 h of 5-min CGM, steps, heart rate and sleep stage (37 summary features) + 24 encoded EHR features (demographics, HbA1c, eGFR, lipids, comorbidities, medication class and dose, prior severe hypo). |
| Outputs | Probability of any CGM > 180 mg/dL (hyper) and any CGM < 70 mg/dL (hypo) in the next 2 h, every 30 min; an alert when risk reaches a threshold set at a 5% false-alert rate on validation patients. |
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

## Performance

Onset windows (the patient is not already past the threshold). 95% CIs resample whole patients.

| Evaluation | Target | AUROC | AUPRC (prevalence) |
|---|---|---|---|
| Synthetic test, 60 patients, deployed model | Hyper | 0.934 (0.923-0.949) | 0.739 (0.156) |
| Synthetic test, 60 patients, deployed model | Hypo | 0.987 (0.979-0.992) | 0.705 (0.022) |
| Synthetic test, nocturnal | Hypo | 0.989 (0.980-0.995) | 0.659 (0.015) |
| **Real CGM (CGMacros T2D, 14), CGM-only model, zero-shot** | Hyper | **0.698 (0.656-0.736)** | **0.463 (0.276)** |
| Real CGM, same features trained on the real cohort (leave-one-out) | Hyper | 0.771 (0.732-0.808) | 0.537 (0.276) |

* **The synthetic numbers are optimistic.** On real CGM, discrimination falls from 0.93 to 0.70 AUROC.
  Simulated lows drift in slowly (the lowest CGM in the 2 h before onset is already ~70 mg/dL), which makes
  them easy to anticipate. Hypoglycaemia could not be checked on real data: only 3 CGMacros T2D
  participants go low.
* **Calibration:** good on synthetic test data (hypo ECE 0.004), poor on real CGM (zero-shot ECE 0.14).
  Thresholds and probabilities would need recalibrating on the target population.
* **EHR value:** +0.051 hypo AUPRC (95% CI +0.018 to +0.113) on synthetic data, partly by construction;
  untested on real data (CGMacros has no medication records).
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
* **Alert fatigue.** At a 10% false-alert rate, only ~18% of window-level hypo alerts are true positives.
* **Automation bias.** Explanations describe what the model reacted to, not physiological causes.
* **Equity.** The synthetic cohort is loosely modelled on Indian T2D patients but is not epidemiology. Region,
  diet, socioeconomic status and device type are absent, and real performance across them is unknown.
* **Privacy.** No real patient data is used. A real deployment would process sensitive health data; see
  [deployment_path.md](deployment_path.md).

## Recommendations

Before any clinical use: validate on real paired EHR + CGM data from the target population, recalibrate,
evaluate per-episode alerting with clinicians, complete the subgroup audit on a representative cohort,
and obtain ethics and regulatory review ([deployment_path.md](deployment_path.md)).
