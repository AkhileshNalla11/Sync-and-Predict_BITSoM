# Subgroup performance (deployed model)

**SYNTHETIC DATA.** LightGBM on CGM + wearables + EHR, 60 test patients, **onset** windows. Subgroup
differences reflect how the simulator generates each group, not real-world disparities, but the check
shows how the model would be audited. Groups with fewer than 20 positive windows or 3 patients with events are
not scored (too few to say anything). 95% CIs resample whole patients. *Lift* = AUPRC / prevalence:
AUPRC depends on prevalence, so lift is the fairer comparison across groups. *Calibration* compares
the mean predicted risk with the observed event rate.

## Hyperglycaemia (> 180 mg/dL in 2 h)

All test patients: AUROC 0.934, AUPRC 0.739, prevalence 0.156, lift 4.7x.

| Dimension | Group | Patients | Positives (patients) | Prevalence | AUROC (95% CI) | AUPRC (95% CI) | Lift | Mean predicted |
|---|---|---|---|---|---|---|---|---|
| Sex | Female | 24 | 1364 (24) | 0.109 | 0.936 (0.920-0.952) | 0.703 (0.597-0.765) | 6.4x | 0.125 |
| Sex | Male | 36 | 3453 (33) | 0.188 | 0.932 (0.917-0.946) | 0.754 (0.708-0.791) | 4.0x | 0.172 |
| Age | 50-64 | 38 | 2787 (37) | 0.146 | 0.935 (0.919-0.950) | 0.722 (0.666-0.776) | 5.0x | 0.152 |
| Age | < 50 | 14 | 1359 (12) | 0.183 | 0.930 (0.910-0.950) | 0.755 (0.663-0.807) | 4.1x | 0.166 |
| Age | >= 65 | 8 | 671 (8) | 0.157 | 0.943 (0.918-0.964) | 0.782 (0.723-0.843) | 5.0x | 0.136 |
| Kidney function (eGFR) | < 60 (reduced) | 1 | 131 (1) | 0.242 | *too few events* | | | 0.167 |
| Kidney function (eGFR) | >= 60 | 59 | 4686 (56) | 0.155 | 0.935 (0.924-0.945) | 0.740 (0.702-0.782) | 4.8x | 0.153 |
| Drug class | Insulin | 7 | 599 (7) | 0.163 | 0.916 (0.889-0.938) | 0.710 (0.593-0.810) | 4.3x | 0.170 |
| Drug class | Neither | 35 | 3113 (33) | 0.179 | 0.927 (0.910-0.945) | 0.746 (0.682-0.788) | 4.2x | 0.175 |
| Drug class | Sulfonylurea (no insulin) | 18 | 1105 (17) | 0.113 | 0.953 (0.932-0.970) | 0.737 (0.653-0.807) | 6.5x | 0.108 |
| HbA1c | 7-8.9% | 39 | 3900 (39) | 0.199 | 0.922 (0.906-0.935) | 0.753 (0.713-0.791) | 3.8x | 0.185 |
| HbA1c | < 7% | 14 | 332 (11) | 0.038 | 0.939 (0.918-0.953) | 0.502 (0.395-0.579) | 13.4x | 0.064 |
| HbA1c | >= 9% | 7 | 585 (7) | 0.246 | 0.931 (0.913-0.943) | 0.781 (0.654-0.875) | 3.2x | 0.220 |
| BMI | < 25 | 21 | 1340 (19) | 0.111 | 0.942 (0.928-0.957) | 0.672 (0.582-0.734) | 6.1x | 0.107 |
| BMI | >= 25 | 39 | 3477 (38) | 0.185 | 0.928 (0.915-0.943) | 0.762 (0.713-0.801) | 4.1x | 0.183 |

## Hypoglycaemia (< 70 mg/dL in 2 h)

All test patients: AUROC 0.987, AUPRC 0.705, prevalence 0.022, lift 31.5x.

| Dimension | Group | Patients | Positives (patients) | Prevalence | AUROC (95% CI) | AUPRC (95% CI) | Lift | Mean predicted |
|---|---|---|---|---|---|---|---|---|
| Sex | Female | 24 | 257 (6) | 0.017 | 0.989 (0.980-0.996) | 0.666 (0.455-0.804) | 39.9x | 0.016 |
| Sex | Male | 36 | 598 (11) | 0.026 | 0.986 (0.977-0.994) | 0.721 (0.533-0.796) | 27.5x | 0.022 |
| Age | 50-64 | 38 | 383 (9) | 0.016 | 0.989 (0.982-0.994) | 0.651 (0.450-0.779) | 41.2x | 0.013 |
| Age | < 50 | 14 | 258 (4) | 0.029 | 0.987 (0.970-0.997) | 0.781 (0.382-0.821) | 26.8x | 0.025 |
| Age | >= 65 | 8 | 214 (4) | 0.042 | 0.977 (0.954-0.993) | 0.700 (0.309-0.854) | 16.6x | 0.039 |
| Kidney function (eGFR) | < 60 (reduced) | 1 | 0 (0) | 0.000 | *too few events* | | | 0.001 |
| Kidney function (eGFR) | >= 60 | 59 | 855 (17) | 0.023 | 0.987 (0.980-0.993) | 0.705 (0.495-0.770) | 31.0x | 0.020 |
| Drug class | Insulin | 7 | 34 (4) | 0.008 | 0.967 (0.943-0.987) | 0.217 (0.142-0.280) | 28.4x | 0.007 |
| Drug class | Neither | 35 | 0 (0) | 0.000 | *too few events* | | | 0.000 |
| Drug class | Sulfonylurea (no insulin) | 18 | 821 (13) | 0.073 | 0.961 (0.940-0.979) | 0.722 (0.606-0.790) | 9.8x | 0.063 |
| HbA1c | 7-8.9% | 39 | 556 (11) | 0.022 | 0.987 (0.979-0.993) | 0.683 (0.500-0.767) | 30.4x | 0.020 |
| HbA1c | < 7% | 14 | 283 (4) | 0.032 | 0.985 (0.973-0.997) | 0.770 (0.427-0.822) | 24.2x | 0.027 |
| HbA1c | >= 9% | 7 | 16 (2) | 0.004 | *too few events* | | | 0.004 |
| BMI | < 25 | 21 | 496 (8) | 0.037 | 0.980 (0.966-0.994) | 0.720 (0.478-0.794) | 19.4x | 0.033 |
| BMI | >= 25 | 39 | 359 (9) | 0.014 | 0.991 (0.985-0.996) | 0.688 (0.386-0.809) | 47.5x | 0.012 |

## Reading this

* **Hyper:** AUROC ranges from 0.916 (Drug class: Insulin) to 0.953 (Drug class: Sulfonylurea (no insulin)) across scored groups.
* **Hypo:** AUROC ranges from 0.961 (Drug class: Sulfonylurea (no insulin)) to 0.991 (BMI: >= 25) across scored groups.
* **Lowest hyper AUPRC:** HbA1c: < 7% (0.502 at prevalence 0.038, lift 13.4x; 14 patients, 332 positive windows from 11 patients). Low AUPRC partly reflects rare events in that group, so read it with AUROC and lift; small groups make all of these unstable.
* **Lowest hypo AUPRC:** Drug class: Insulin (0.217 at prevalence 0.008, lift 28.4x; 7 patients, 34 positive windows from 4 patients). Low AUPRC partly reflects rare events in that group, so read it with AUROC and lift; small groups make all of these unstable.
* **Under-represented in the test set:** Kidney function (eGFR) < 60 (reduced) (1 patient). These groups cannot be audited at all here, even though the simulator gives them higher hypo risk by design (e.g. slower drug clearance with low eGFR).
* **Hypoglycaemia cannot be audited in every group:** Kidney function (eGFR) < 60 (reduced), Drug class Neither, HbA1c >= 9% have too few events. In this simulator, patients on neither insulin nor a sulfonylurea essentially never go low, which is clinically expected.
* A real audit would need a much larger, representative cohort, and should add region, diet, socioeconomic status and device type, which this synthetic EHR does not contain.
