# Stage 4 fusion model results

Test patients only. 95% CIs from a patient-level bootstrap. **SYNTHETIC DATA, not clinically validated.**

*onset* = patient not already past the threshold at prediction time (headline); *all* = every window.
FAR = false-alert rate (fraction of negative windows that trigger an alert). Alert burden counts
alerting 30-min windows per monitored patient-day (nocturnal rows: per patient-night), not episodes.

| Target | Subset | Model | Prevalence | AUPRC | AUROC | Sens @ 10% FAR | Alerting windows / patient-day @ 10% FAR |
|---|---|---|---|---|---|---|---|
| hyper | onset | fusion_gru | 0.156 | 0.740 (0.695-0.775) | 0.934 (0.922-0.949) | 0.765 | 7.8 |
| hyper | onset | seqonly_gru | 0.156 | 0.719 (0.677-0.755) | 0.925 (0.914-0.938) | 0.747 | 7.7 |
| hyper | onset | fusion_cnn | 0.156 | 0.726 (0.678-0.765) | 0.931 (0.918-0.945) | 0.760 | 7.8 |
| hyper | all | fusion_gru | 0.323 | 0.951 (0.929-0.966) | 0.972 (0.963-0.979) | 0.900 | 17.2 |
| hyper | all | seqonly_gru | 0.323 | 0.948 (0.927-0.964) | 0.969 (0.960-0.976) | 0.898 | 17.2 |
| hyper | all | fusion_cnn | 0.323 | 0.950 (0.928-0.966) | 0.971 (0.962-0.978) | 0.901 | 17.2 |
| hypo | onset | fusion_gru | 0.022 | 0.672 (0.508-0.778) | 0.985 (0.976-0.992) | 0.980 | 5.7 |
| hypo | onset | seqonly_gru | 0.022 | 0.662 (0.496-0.762) | 0.984 (0.976-0.990) | 0.973 | 5.7 |
| hypo | onset | fusion_cnn | 0.022 | 0.667 (0.523-0.769) | 0.985 (0.978-0.991) | 0.979 | 5.7 |
| hypo | all | fusion_gru | 0.035 | 0.838 (0.711-0.901) | 0.990 (0.982-0.995) | 0.985 | 6.3 |
| hypo | all | seqonly_gru | 0.035 | 0.832 (0.694-0.894) | 0.989 (0.982-0.994) | 0.983 | 6.3 |
| hypo | all | fusion_cnn | 0.035 | 0.837 (0.710-0.900) | 0.990 (0.984-0.995) | 0.987 | 6.3 |
| hypo | nocturnal_onset | fusion_gru | 0.015 | 0.622 (0.231-0.822) | 0.986 (0.973-0.993) | 0.984 | 5.4 |
| hypo | nocturnal_onset | seqonly_gru | 0.015 | 0.619 (0.210-0.828) | 0.989 (0.980-0.994) | 0.989 | 5.4 |
| hypo | nocturnal_onset | fusion_cnn | 0.015 | 0.617 (0.339-0.818) | 0.989 (0.977-0.995) | 1.000 | 5.4 |
| hypo | nocturnal | fusion_gru | 0.022 | 0.777 (0.251-0.921) | 0.990 (0.975-0.996) | 0.989 | 5.7 |
| hypo | nocturnal | seqonly_gru | 0.022 | 0.783 (0.244-0.920) | 0.992 (0.983-0.996) | 0.993 | 5.7 |
| hypo | nocturnal | fusion_cnn | 0.022 | 0.788 (0.385-0.925) | 0.992 (0.980-0.996) | 1.000 | 5.7 |
