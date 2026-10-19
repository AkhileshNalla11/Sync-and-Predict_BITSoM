"""External sanity check on real CGM data (CGMacros, PhysioNet). Does not touch the synthetic pipeline.

    python -m src.eval.external_cgmacros --data-dir data/external/cgmacros [--models-dir ...] [--no-lopo]

Written to docs/: external_validation.csv (metrics), external_validation_paired.csv (paired model
differences, T2D hyper) and external_validation_cohort.csv (aggregated cohort description). Models:
  * persistence           trend extrapolation, no training
  * lgbm_cgm_zero_shot    the CGM-only LightGBM trained on SYNTHETIC patients, applied unchanged
  * lgbm_cgm_lopo         CGM-only LightGBM retrained on CGMacros T2D, leave-one-participant-out
                          (hyper only: there are too few real hypo windows to train on)
  * lgbm_cgm_zero_shot_platt / _isotonic
                          the zero-shot scores recalibrated on real data, leave-one-participant-out: each
                          participant's probabilities are mapped by a calibrator fitted on the OTHER T2D
                          participants' onset windows (Platt: logistic on the logit; isotonic regression).
                          The model itself is unchanged; Platt keeps the ranking (AUROC) exactly.
Only the "cgm" + "time" feature groups transfer; CGMacros has no EHR in our format and the Fitbit
stream is not mapped onto our wearable features. See docs/external_validation.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.config import ROOT, load_config, resolve_path
from src.data.cgmacros import DEFAULT_SENSOR, STATUSES, build_windows, cohort_table, load_cohort
from src.eval.metrics import auprc, auroc, calibration_summary, evaluate_targets, paired_bootstrap_diff
from src.models.baselines import FEATURE_SETS, TARGETS, persistence_scores, train_lgbm

FEATURE_SET = "cgm"
LOPO_INNER_VAL = 3            # participants held out of each LOPO training fold for early stopping
MIN_POSITIVE_WINDOWS = 100    # train a LOPO model for a target only if the cohort has this many positives


def load_zero_shot(models_dir: Path) -> dict[str, lgb.Booster]:
    boosters = {t: lgb.Booster(model_file=str(models_dir / f"lgbm_{FEATURE_SET}_{t}.txt")) for t in TARGETS}
    for t, b in boosters.items():
        if b.feature_name() != list(FEATURE_SETS[FEATURE_SET]):
            raise ValueError(f"lgbm_{FEATURE_SET}_{t}: feature names do not match FEATURE_SETS[{FEATURE_SET!r}]")
    return boosters


def lopo_scores(windows: pd.DataFrame, X: pd.DataFrame, target: str, cfg: dict[str, Any]) -> np.ndarray:
    """Out-of-fold scores: each participant is scored by a model that never saw them.

    Within each fold, LOPO_INNER_VAL other participants are held out for early stopping.
    Scores from different folds are pooled for evaluation, as is usual for LOPO.
    """
    y = windows[target].to_numpy(int)
    pids = windows.patient_id.to_numpy()
    uniq = np.unique(pids)
    out = np.full(len(windows), np.nan)
    for i, p in enumerate(uniq):
        rest = uniq[uniq != p]
        val = np.random.default_rng([cfg["seed"], i]).choice(rest, LOPO_INNER_VAL, replace=False)
        tr, va, te = ~np.isin(pids, np.append(val, p)), np.isin(pids, val), pids == p
        booster = train_lgbm(X[tr], y[tr], X[va], y[va], cfg)
        out[te] = booster.predict(X[te], num_iteration=booster.best_iteration)
    return out


def lopo_recalibrate(windows: pd.DataFrame, p: np.ndarray, target: str, method: str) -> np.ndarray:
    """Out-of-fold recalibration of fixed scores. For each participant, a calibrator is fitted on the other
    participants' onset windows (the windows alerts are raised on) and applied to all of this participant's."""
    y = windows[target].to_numpy(int)
    onset = ~windows[f"in_event_{target}"].to_numpy(bool)
    pids = windows.patient_id.to_numpy()
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    out = np.full(len(p), np.nan)
    for pid in np.unique(pids):
        fit, te = onset & (pids != pid), pids == pid
        if method == "platt":
            out[te] = LogisticRegression(C=1e4).fit(z[fit], y[fit]).predict_proba(z[te])[:, 1]
        else:
            out[te] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p[fit], y[fit]).predict(p[te])
    return out


def calibration_cols(windows: pd.DataFrame, scores: dict[str, np.ndarray], target: str, subset: str) -> dict[str, float]:
    """Brier / ECE / mean predicted probability on the onset and all-window subsets."""
    if subset not in ("all", "onset"):
        return {}
    m = np.ones(len(windows), bool) if subset == "all" else ~windows[f"in_event_{target}"].to_numpy(bool)
    c = calibration_summary(windows[target].to_numpy()[m], scores[target][m])
    return {"brier": c["brier"], "ece": c["ece"], "mean_predicted": c["mean_predicted"]}


PAIRS = (("lgbm_cgm_zero_shot", "persistence"), ("lgbm_cgm_lopo", "persistence"), ("lgbm_cgm_lopo", "lgbm_cgm_zero_shot"))
RECAL = ("platt", "isotonic")


def paired_rows(w: pd.DataFrame, models: dict[str, dict[str, np.ndarray]], target: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Paired patient-bootstrap differences (A - B) in AUPRC and AUROC on onset and all windows."""
    rows = []
    for subset in ("onset", "all"):
        m = np.ones(len(w), bool) if subset == "all" else ~w[f"in_event_{target}"].to_numpy(bool)
        y, g = w[target].to_numpy(int)[m], w.patient_id.to_numpy()[m]
        for a, b in PAIRS:
            if a not in models or b not in models:
                continue
            for metric, fn in (("auprc", auprc), ("auroc", auroc)):
                d = paired_bootstrap_diff(y, models[a][target][m], models[b][target][m], g, fn,
                                          cfg["eval"]["n_bootstrap"], cfg["seed"])
                rows.append({"target": target, "subset": subset, "model_a": a, "model_b": b, "metric": metric, **d})
    return rows


def run(cfg: dict[str, Any], data_dir: Path, models_dir: Path, sensor: str,
        lopo: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = load_cohort(data_dir, sensor=sensor, statuses=STATUSES, dt_min=cfg["sampling"]["interval_minutes"])
    windows, feats = build_windows(cohort, cfg)
    X = feats[list(FEATURE_SETS[FEATURE_SET])]
    horizon_min = cfg["windows"]["horizon_hours"] * 60
    boosters = load_zero_shot(models_dir)

    # cohort summary (aggregated; no per-participant rows leave data/external)
    ct = cohort_table(cohort)
    g = windows.groupby("status")
    summary = ct.groupby("status").agg(participants=("participant_id", "size"), hba1c_median=("hba1c", "median"),
                                       days=("days", "sum"), cgm_coverage=("coverage", "mean"),
                                       gaps_ge_30min=("n_gaps_ge_30min", "sum"), longest_gap_h=("longest_gap_h", "max"),
                                       bridged_minutes_masked=("bridged_minutes_masked", "sum"),
                                       out_of_range_readings=("out_of_range_readings", "sum"))
    summary["windows"] = g.size()
    for t in TARGETS:
        onset = ~windows[f"in_event_{t}"]
        summary[f"{t}_windows"] = g[t].sum()
        summary[f"{t}_onset_windows"] = (windows[t] & onset).groupby(windows.status).sum()
        summary[f"{t}_onset_prevalence"] = (windows[t][onset]).groupby(windows.status[onset]).mean()
        summary[f"participants_with_{t}"] = windows[windows[t]].groupby("status").patient_id.nunique()
    summary = summary.reindex(list(STATUSES)).fillna(0).reset_index()

    rows, paired = [], []
    for status in STATUSES:
        m = (windows.status == status).to_numpy()
        if not m.any():
            continue
        w = windows[m].reset_index(drop=True)
        models = {"persistence": persistence_scores(feats[m], horizon_min),
                  "lgbm_cgm_zero_shot": {t: b.predict(X[m]) for t, b in boosters.items()}}
        if status == "t2d":
            zs = models["lgbm_cgm_zero_shot"]
            for method in RECAL:
                models[f"lgbm_cgm_zero_shot_{method}"] = {
                    t: lopo_recalibrate(w, zs[t], t, method) if w[t].sum() >= MIN_POSITIVE_WINDOWS else np.zeros(len(w))
                    for t in TARGETS}
        if lopo and status == "t2d":
            models["lgbm_cgm_lopo"] = {t: (lopo_scores(w, X[m].reset_index(drop=True), t, cfg)
                                           if w[t].sum() >= MIN_POSITIVE_WINDOWS else np.zeros(len(w)))
                                       for t in TARGETS}
        for name, scores in models.items():
            for r in evaluate_targets(w, scores, cfg):
                fitted_here = name == "lgbm_cgm_lopo" or name.startswith("lgbm_cgm_zero_shot_")
                if fitted_here and w[r["target"]].sum() < MIN_POSITIVE_WINDOWS:
                    continue                               # not trained / recalibrated for this target
                extra = calibration_cols(w, scores, r["target"], r["subset"]) if name != "persistence" else {}
                rows.append({"cohort": status, "model": name, **r, **extra})
            print(f"  [{status}] {name} done", flush=True)
        if status == "t2d":
            paired += [{"cohort": status, **r} for r in paired_rows(w, models, "hyper", cfg)]
    return pd.DataFrame(rows), pd.DataFrame(paired), summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot + LOPO sanity check on CGMacros")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "external" / "cgmacros"))
    ap.add_argument("--models-dir", default=None, help="defaults to paths.models_dir in config.yaml")
    ap.add_argument("--sensor", default=DEFAULT_SENSOR, choices=("dexcom", "libre"))
    ap.add_argument("--no-lopo", action="store_true")
    ap.add_argument("--out-dir", default=None, help="defaults to paths.docs_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    models_dir = Path(args.models_dir) if args.models_dir else resolve_path(cfg["paths"]["models_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else resolve_path(cfg["paths"]["docs_dir"])
    res, paired, summary = run(cfg, Path(args.data_dir), models_dir, args.sensor, not args.no_lopo)
    suffix = "" if args.sensor == DEFAULT_SENSOR else f"_{args.sensor}"
    res.to_csv(out_dir / f"external_validation{suffix}.csv", index=False)
    paired.to_csv(out_dir / f"external_validation_paired{suffix}.csv", index=False)
    summary.to_csv(out_dir / f"external_validation_cohort{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    cols = ["cohort", "target", "subset", "model", "n_patients", "n_positive", "prevalence", "auprc", "auprc_lo",
            "auprc_hi", "auroc", "auroc_lo", "auroc_hi", "sens_at_far10", "ppv_at_far10"]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    cal = res.dropna(subset=["brier"])[["cohort", "target", "subset", "model", "prevalence", "mean_predicted", "brier", "ece"]]
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(paired.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
