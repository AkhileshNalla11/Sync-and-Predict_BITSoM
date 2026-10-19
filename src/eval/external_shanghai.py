"""Real linked-data check on ShanghaiT2DM: does the EHR add value when it truly belongs to the CGM's patient?

    python -m src.eval.external_shanghai --data-dir data/external/shanghai

Everything else in this project links EHR and CGM synthetically. ShanghaiT2DM (src/data/shanghai.py) has
both for the same 100 real patients, so the project's central question can be asked on real data:
  * persistence           trend extrapolation, no training
  * lgbm_cgm_zero_shot    the CGM-only LightGBM trained on SYNTHETIC patients, applied unchanged
  * Trained on ShanghaiT2DM with patient-grouped 10-fold cross-validation (every patient is scored by a
    model that never saw any of their periods; early stopping on held-out training patients):
      lgbm_cgm             CGM + time of day
      lgbm_cgm_ehr         + this project's 24 EHR features, mapped from the real record
      lgbm_cgm_ehr_doses   + time-varying features from the logged insulin / oral drugs / meals
      lgbm_cgm_full_ehr    + the extra clinical fields a real record has (C-peptide, glycated albumin, ...)
  * A low-capacity check, because trees can memorise 100 patients from their static features: logistic
    regression on the out-of-fold lgbm_cgm logit alone (stack_cgm) vs the same plus 8 core EHR fields
    (stack_cgm_ehr), fitted on the training folds. If the EHR carries signal the CGM lacks, this shows it.
All folds are shared, so model differences are paired by patient. Written to docs/: external_shanghai.csv
(metrics), external_shanghai_paired.csv (paired differences), external_shanghai_cohort.csv (aggregate
cohort description) and external_shanghai_importance.csv (gain share by data stream). Per-patient rows
never leave data/external.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import ROOT, load_config, resolve_path
from src.data.ehr_features import STATIC_FEATURES
from src.data.shanghai import DOSE_FEATURES, EXTRA_STATIC, Period, dose_features, load_cohort
from src.data.wearable_features import SERIES_CHANNELS, build_feature_table, label_windows
from src.eval.external_cgmacros import calibration_cols, load_zero_shot
from src.eval.metrics import auprc, auroc, evaluate_targets, paired_bootstrap_diff
from src.models.baselines import FEATURE_SETS, TARGETS, persistence_scores, train_lgbm

N_FOLDS = 10
INNER_VAL_FRACTION = 0.15
CGM = tuple(FEATURE_SETS["cgm"])
VARIANTS: dict[str, tuple[str, ...]] = {
    "lgbm_cgm": CGM,
    "lgbm_cgm_ehr": CGM + tuple(STATIC_FEATURES),
    "lgbm_cgm_ehr_doses": CGM + tuple(STATIC_FEATURES) + DOSE_FEATURES,
    "lgbm_cgm_full_ehr": CGM + tuple(STATIC_FEATURES) + EXTRA_STATIC,
}
PAIRS = (("lgbm_cgm_ehr", "lgbm_cgm"), ("lgbm_cgm_ehr_doses", "lgbm_cgm_ehr"), ("lgbm_cgm_ehr_doses", "lgbm_cgm"),
         ("lgbm_cgm_full_ehr", "lgbm_cgm"), ("lgbm_cgm", "persistence"), ("lgbm_cgm_zero_shot", "persistence"),
         ("lgbm_cgm", "lgbm_cgm_zero_shot"), ("stack_cgm_ehr", "stack_cgm"))
STACK_EHR = ("on_insulin", "on_sulfonylurea", "on_metformin", "hba1c", "egfr", "age", "bmi", "diabetes_duration_years")


def build_windows(periods: list[Period], static: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Labelled windows (unchanged label_windows) and features: CGM + time, EHR per period, logged doses."""
    dt = cfg["sampling"]["interval_minutes"]
    win_parts, feat_parts = [], []
    for i, p in enumerate(periods):
        ts = pd.DatetimeIndex(p.cgm.index)
        cgm = p.cgm.to_numpy(np.float32)[None, :]
        w = label_windows(cgm, ts, cfg)
        if w.empty:
            continue
        series = np.full((1, len(ts), len(SERIES_CHANNELS)), np.nan, np.float32)
        series[0, :, 0] = cgm[0]
        f = build_feature_table(series, w, ts, cfg)
        f = pd.concat([f.reset_index(drop=True),
                       static.loc[[p.period_id] * len(w)].reset_index(drop=True),
                       dose_features(p.doses, w.anchor_idx.to_numpy(), dt)], axis=1)
        win_parts.append(w.assign(patient_idx=i, period_id=p.period_id, patient_id=p.patient_id,
                                  timestamp=ts[w.anchor_idx.to_numpy()]))
        feat_parts.append(f)
    return pd.concat(win_parts, ignore_index=True), pd.concat(feat_parts, ignore_index=True)


def folds(patients: np.ndarray, seed: int) -> list[np.ndarray]:
    """Patient ids split into N_FOLDS groups (all periods of a patient stay together)."""
    uniq = np.random.default_rng(seed).permutation(np.unique(patients))
    return [uniq[k::N_FOLDS] for k in range(N_FOLDS)]


def cv_scores(windows: pd.DataFrame, X: pd.DataFrame, target: str, cfg: dict[str, Any],
              gain: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Out-of-fold scores with patient-grouped folds; optionally accumulates feature gain."""
    y = windows[target].to_numpy(int)
    pids = windows.patient_id.to_numpy()
    out = np.full(len(windows), np.nan)
    for k, test_pts in enumerate(folds(pids, cfg["seed"])):
        rest = np.setdiff1d(np.unique(pids), test_pts)
        n_val = max(5, int(round(INNER_VAL_FRACTION * len(rest))))
        val_pts = np.random.default_rng([cfg["seed"], k]).choice(rest, n_val, replace=False)
        te, va = np.isin(pids, test_pts), np.isin(pids, val_pts)
        tr = ~te & ~va
        b = train_lgbm(X[tr], y[tr], X[va], y[va], cfg)
        out[te] = b.predict(X[te], num_iteration=b.best_iteration)
        if gain is not None:
            g = b.feature_importance("gain", iteration=b.best_iteration)
            gain[target] = gain.get(target, 0) + g / max(g.sum(), 1e-12)
    return out


def stacked_scores(windows: pd.DataFrame, feats: pd.DataFrame, cgm_oof: np.ndarray, target: str,
                   cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold logistic regression on the CGM model's logit, without and with 8 core EHR fields."""
    p = np.clip(cgm_oof, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))[:, None]
    e = feats[list(STACK_EHR)].astype(float)
    e = e.fillna(e.median())
    e = ((e - e.mean()) / e.std(ddof=0).replace(0, 1)).to_numpy()
    y = windows[target].to_numpy(int)
    pids = windows.patient_id.to_numpy()
    base, with_ehr = np.zeros(len(y)), np.zeros(len(y))
    for test_pts in folds(pids, cfg["seed"]):
        te = np.isin(pids, test_pts)
        base[te] = LogisticRegression(max_iter=1000).fit(z[~te], y[~te]).predict_proba(z[te])[:, 1]
        x = np.hstack([z, e])
        with_ehr[te] = LogisticRegression(max_iter=1000).fit(x[~te], y[~te]).predict_proba(x[te])[:, 1]
    return base, with_ehr


def stream_of(feature: str) -> str:
    if feature in DOSE_FEATURES:
        return "Logged doses / meals"
    if feature in EXTRA_STATIC:
        return "EHR, extra fields"
    if feature in STATIC_FEATURES:
        return "EHR, project's 24 features"
    return "CGM + time"


def cohort_summary(periods: list[Period], static: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    r = pd.DataFrame([{**p.report, "period_id": p.period_id} for p in periods]).set_index("period_id")
    st = static.loc[r.index]
    rows = {"patients": windows.patient_id.nunique(), "periods": len(r), "monitored_days": r.days.sum(),
            "median_days_per_period": r.days.median(), "cgm_coverage_after_interpolation": r.coverage.mean(),
            "readings_clipped_to_40_400": int(r.clipped_readings.sum()),
            "age_median": st.age.median(), "hba1c_median_pct": st.hba1c.median(), "egfr_median": st.egfr.median(),
            "share_on_insulin": st.on_insulin.mean(), "share_on_sulfonylurea": st.on_sulfonylurea.mean(),
            "share_on_metformin": st.on_metformin.mean(), "windows": len(windows)}
    for t in TARGETS:
        onset = ~windows[f"in_event_{t}"]
        rows[f"{t}_onset_windows"] = int((windows[t] & onset).sum())
        rows[f"{t}_onset_prevalence"] = float(windows[t][onset].mean())
        rows[f"patients_with_{t}"] = int(windows[windows[t]].patient_id.nunique())
    noct = ~windows.in_event_hypo & windows.nocturnal
    rows["nocturnal_hypo_onset_windows"] = int((windows.hypo & noct).sum())
    return pd.DataFrame([rows]).T.reset_index().rename(columns={"index": "field", 0: "value"})


def subgroup_rows(w: pd.DataFrame, models: dict[str, dict[str, np.ndarray]], static: pd.DataFrame) -> list[dict[str, Any]]:
    """Hypo onset AUPRC / AUROC for insulin users vs non-users (the project's weakest synthetic subgroup)."""
    rows = []
    on_ins = static.on_insulin.reindex(w.period_id).to_numpy() == 1
    onset = ~w.in_event_hypo.to_numpy(bool)
    for name, flag in (("on insulin", on_ins), ("not on insulin", ~on_ins)):
        m = onset & flag
        y = w.hypo.to_numpy(int)[m]
        for model, sc in models.items():
            rows.append({"target": "hypo", "subset": f"onset, {name}", "model": model, "n_patients": int(w.patient_id[m].nunique()),
                         "n_positive": int(y.sum()), "prevalence": float(y.mean()),
                         "auprc": auprc(y, sc["hypo"][m]), "auroc": auroc(y, sc["hypo"][m])})
    return rows


def run(cfg: dict[str, Any], data_dir: Path, models_dir: Path) -> dict[str, pd.DataFrame]:
    periods, static, _ = load_cohort(data_dir, cfg["sampling"]["interval_minutes"])
    w, feats = build_windows(periods, static, cfg)
    print(f"{len(periods)} periods, {w.patient_id.nunique()} patients, {len(w):,} windows", flush=True)
    horizon_min = cfg["windows"]["horizon_hours"] * 60
    models: dict[str, dict[str, np.ndarray]] = {
        "persistence": persistence_scores(feats, horizon_min),
        "lgbm_cgm_zero_shot": {t: b.predict(feats[list(CGM)]) for t, b in load_zero_shot(models_dir).items()},
    }
    gains: dict[str, dict[str, np.ndarray]] = {}
    for name, cols in VARIANTS.items():
        gains[name] = {}
        models[name] = {t: cv_scores(w, feats[list(cols)], t, cfg, gains[name]) for t in TARGETS}
        print(f"  {name} done", flush=True)
    stacks = {t: stacked_scores(w, feats, models["lgbm_cgm"][t], t, cfg) for t in TARGETS}
    models["stack_cgm"] = {t: v[0] for t, v in stacks.items()}
    models["stack_cgm_ehr"] = {t: v[1] for t, v in stacks.items()}

    rows, paired = [], []
    for name, scores in models.items():
        for r in evaluate_targets(w, scores, cfg):
            extra = calibration_cols(w, scores, r["target"], r["subset"]) if name != "persistence" else {}
            rows.append({"model": name, **r, **extra})
    for t in TARGETS:
        for subset in ("onset", "nocturnal_onset") if t == "hypo" else ("onset",):
            m = ~w[f"in_event_{t}"].to_numpy(bool)
            if subset.startswith("nocturnal"):
                m &= w.nocturnal.to_numpy(bool)
            y, g = w[t].to_numpy(int)[m], w.patient_id.to_numpy()[m]
            for a, b in PAIRS:
                for metric, fn in (("auprc", auprc), ("auroc", auroc)):
                    d = paired_bootstrap_diff(y, models[a][t][m], models[b][t][m], g, fn, cfg["eval"]["n_bootstrap"], cfg["seed"])
                    paired.append({"target": t, "subset": subset, "model_a": a, "model_b": b, "metric": metric, **d})
    imp = []
    for name, by_t in gains.items():
        cols = VARIANTS[name]
        for t, g in by_t.items():
            share = pd.Series(g / N_FOLDS, index=list(cols)).groupby(stream_of).sum()
            imp += [{"model": name, "target": t, "stream": s, "gain_share": v} for s, v in share.items()]
            top = pd.Series(g / N_FOLDS, index=list(cols))
            top = top[[c for c in cols if stream_of(c) != "CGM + time"]].sort_values(ascending=False).head(5)
            imp += [{"model": name, "target": t, "stream": f"top non-CGM feature: {f}", "gain_share": v} for f, v in top.items()]
    return {"metrics": pd.concat([pd.DataFrame(rows), pd.DataFrame(subgroup_rows(w, models, static))], ignore_index=True),
            "paired": pd.DataFrame(paired), "cohort": cohort_summary(periods, static, w), "importance": pd.DataFrame(imp)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Real linked EHR + CGM check on ShanghaiT2DM")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "external" / "shanghai"))
    ap.add_argument("--models-dir", default=None, help="defaults to paths.models_dir in config.yaml")
    ap.add_argument("--out-dir", default=None, help="defaults to paths.docs_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    models_dir = Path(args.models_dir) if args.models_dir else resolve_path(cfg["paths"]["models_dir"])
    out = Path(args.out_dir) if args.out_dir else resolve_path(cfg["paths"]["docs_dir"])
    res = run(cfg, Path(args.data_dir), models_dir)
    for key, df in res.items():
        df.to_csv(out / ("external_shanghai.csv" if key == "metrics" else f"external_shanghai_{key}.csv"), index=False)
    pd.set_option("display.width", 250)
    m = res["metrics"]
    cols = ["target", "subset", "model", "n_patients", "n_positive", "prevalence", "auprc", "auprc_lo", "auprc_hi",
            "auroc", "auroc_lo", "auroc_hi", "ece"]
    print(m[[c for c in cols if c in m]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(res["paired"].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(res["cohort"].to_string(index=False))
    print(res["importance"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
