# Demo bundle

**SYNTHETIC DATA. Not for clinical use.** Minimal files for running the dashboard without the
pipeline (e.g. on Streamlit Community Cloud). Rebuild with `python scripts/make_demo_bundle.py`
after re-running the pipeline.

- Built from commit `45d86e6`, seed 42, model `lgbm_cgm_wearables_ehr`
- Total size 14.5 MB

| File | Bytes | SHA-256 (first 16) |
|---|---|---|
| raw/ehr.csv | 42,698 | `41b5ccb4d7d03918` |
| processed/series.npz | 4,346,261 | `e003e039d8672694` |
| processed/windows.csv.gz | 1,220,951 | `9b8a0359d24e9065` |
| processed/static_features.csv | 88,134 | `9ced834a9acb6841` |
| processed/static_encoder.json | 1,033 | `8504ce7a3f139f00` |
| processed/splits.csv | 6,278 | `df46b9224cdbe83a` |
| processed/window_features.npz | 7,201,048 | `87ba45b76cb8a988` |
| processed/dataset_summary.json | 1,003 | `0cdf2aa621124dac` |
| processed/linkage_report.json | 213 | `3e4aa6a00581cf7f` |
| processed/patient_log.csv.gz | 126,080 | `4fad2d39c45d13c0` |
| processed/twin_fits.csv | 32,188 | `2f384a0aea705e15` |
| processed/models/lgbm_cgm_wearables_ehr_hyper.txt | 960,438 | `1442d92996052e9f` |
| processed/models/lgbm_cgm_wearables_ehr_hypo.txt | 450,834 | `88200651f428c1da` |
| processed/models/lgbm_cgm_wearables_ehr_alert_thresholds.json | 160 | `bd42e53c015baa94` |
| processed/models/twin_intervals.json | 8,508 | `8bd9afec1a0125eb` |
