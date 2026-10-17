# Stage 3 baseline results

Test patients only. 95% CIs from a patient-level bootstrap. **SYNTHETIC DATA, not clinically validated.**

*onset* = patient not already past the threshold at prediction time (headline); *all* = every window.
FAR = false-alert rate (fraction of negative windows that trigger an alert). Alert burden counts
alerting 30-min windows per monitored patient-day (nocturnal rows: per patient-night), not episodes.

| Target | Subset | Model | Prevalence | AUPRC | AUROC | Sens @ 10% FAR | Alerting windows / patient-day @ 10% FAR |
|---|---|---|---|---|---|---|---|
| hyper | onset | persistence | 0.156 | 0.377 (0.331-0.414) | 0.742 (0.714-0.771) | 0.329 | 5.2 |
| hyper | onset | lgbm_cgm | 0.156 | 0.724 (0.679-0.757) | 0.930 (0.919-0.943) | 0.757 | 7.7 |
| hyper | onset | lgbm_cgm_wearables | 0.156 | 0.721 (0.678-0.754) | 0.928 (0.918-0.940) | 0.747 | 7.7 |
| hyper | onset | lgbm_cgm_wearables_ehr | 0.156 | 0.739 (0.695-0.772) | 0.934 (0.923-0.949) | 0.773 | 7.8 |
| hyper | all | persistence | 0.323 | 0.816 (0.759-0.863) | 0.882 (0.857-0.902) | 0.734 | 14.6 |
| hyper | all | lgbm_cgm | 0.323 | 0.950 (0.930-0.966) | 0.971 (0.963-0.978) | 0.899 | 17.2 |
| hyper | all | lgbm_cgm_wearables | 0.323 | 0.949 (0.929-0.965) | 0.970 (0.962-0.977) | 0.897 | 17.1 |
| hyper | all | lgbm_cgm_wearables_ehr | 0.323 | 0.953 (0.931-0.967) | 0.973 (0.964-0.979) | 0.909 | 17.3 |
| hypo | onset | persistence | 0.022 | 0.192 (0.099-0.295) | 0.907 (0.891-0.922) | 0.602 | 5.3 |
| hypo | onset | lgbm_cgm | 0.022 | 0.649 (0.489-0.743) | 0.981 (0.972-0.989) | 0.964 | 5.6 |
| hypo | onset | lgbm_cgm_wearables | 0.022 | 0.654 (0.484-0.756) | 0.982 (0.974-0.989) | 0.971 | 5.7 |
| hypo | onset | lgbm_cgm_wearables_ehr | 0.022 | 0.705 (0.568-0.782) | 0.987 (0.979-0.992) | 0.980 | 5.7 |
| hypo | all | persistence | 0.035 | 0.272 (0.148-0.400) | 0.924 (0.909-0.937) | 0.749 | 5.9 |
| hypo | all | lgbm_cgm | 0.035 | 0.832 (0.702-0.893) | 0.988 (0.980-0.993) | 0.976 | 6.3 |
| hypo | all | lgbm_cgm_wearables | 0.035 | 0.835 (0.696-0.895) | 0.988 (0.982-0.993) | 0.981 | 6.3 |
| hypo | all | lgbm_cgm_wearables_ehr | 0.035 | 0.856 (0.741-0.904) | 0.991 (0.985-0.995) | 0.987 | 6.3 |
| hypo | nocturnal_onset | persistence | 0.015 | 0.087 (0.024-0.160) | 0.897 (0.871-0.923) | 0.505 | 5.1 |
| hypo | nocturnal_onset | lgbm_cgm | 0.015 | 0.605 (0.259-0.806) | 0.987 (0.976-0.994) | 0.984 | 5.4 |
| hypo | nocturnal_onset | lgbm_cgm_wearables | 0.015 | 0.614 (0.244-0.815) | 0.987 (0.977-0.994) | 0.984 | 5.4 |
| hypo | nocturnal_onset | lgbm_cgm_wearables_ehr | 0.015 | 0.659 (0.337-0.820) | 0.989 (0.980-0.995) | 0.984 | 5.4 |
| hypo | nocturnal | persistence | 0.022 | 0.135 (0.026-0.265) | 0.912 (0.889-0.930) | 0.610 | 5.3 |
| hypo | nocturnal | lgbm_cgm | 0.022 | 0.770 (0.324-0.904) | 0.990 (0.979-0.996) | 0.989 | 5.7 |
| hypo | nocturnal | lgbm_cgm_wearables | 0.022 | 0.772 (0.304-0.909) | 0.991 (0.980-0.996) | 0.989 | 5.7 |
| hypo | nocturnal | lgbm_cgm_wearables_ehr | 0.022 | 0.795 (0.409-0.912) | 0.992 (0.982-0.997) | 0.989 | 5.7 |
