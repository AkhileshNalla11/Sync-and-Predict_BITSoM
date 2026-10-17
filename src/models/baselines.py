"""Stage 3 baselines: persistence extrapolation and per-target LightGBM on handcrafted window features.

    python -m src.models.baselines        # trains on train patients, early-stops on val patients,
                                          # evaluates on test patients

Feature sets (the same code serves the Stage 5 ablation):
    cgm                 CGM summary features + time of day          <- the Stage 3 "CGM-only" baseline
    cgm_wearables       + steps / heart rate / sleep
    cgm_wearables_ehr   + the static EHR vector (simple early fusion by concatenation)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.build_dataset import load_processed
from src.data.ehr_features import STATIC_FEATURES
from src.data.wearable_features import FEATURE_GROUPS, build_feature_table
from src.eval.metrics import evaluate_targets

TARGETS = ("hyper", "hypo")
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "cgm": FEATURE_GROUPS["cgm"] + FEATURE_GROUPS["time"],
    "cgm_wearables": FEATURE_GROUPS["cgm"] + FEATURE_GROUPS["time"] + FEATURE_GROUPS["wearables"],
    "cgm_wearables_ehr": FEATURE_GROUPS["cgm"] + FEATURE_GROUPS["time"] + FEATURE_GROUPS["wearables"] + STATIC_FEATURES,
}


# --------------------------------------------------------------------------- persistence
def persistence_scores(feats: pd.DataFrame, horizon_min: float) -> dict[str, np.ndarray]:
    """Extrapolate the latest CGM value along its 30-min trend over the horizon.

    Scores are the extremes of the straight-line trajectory: its maximum for hyper and minus its
    minimum for hypo, so a higher score always means more risk.
    """
    last = feats.cgm_last.to_numpy(dtype=float)
    slope = np.nan_to_num(feats.cgm_slope_30m.to_numpy(dtype=float), nan=0.0)
    end = last + slope * horizon_min
    return {"hyper": np.maximum(last, end), "hypo": -np.minimum(last, end)}


# --------------------------------------------------------------------------- LightGBM
def feature_matrix(feats: pd.DataFrame, windows: pd.DataFrame, static: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Join window features with the patient's static vector and select one feature set's columns."""
    cols = FEATURE_SETS[feature_set]
    if any(c in STATIC_FEATURES for c in cols):
        st = static.loc[windows.patient_id.to_numpy()].reset_index(drop=True)
        feats = pd.concat([feats.reset_index(drop=True), st], axis=1)
    return feats[list(cols)]


def lgbm_params(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["models"]["lgbm"]
    seed = cfg["seed"]
    return {
        "objective": "binary", "metric": "average_precision", "verbosity": -1,
        "learning_rate": p["learning_rate"], "num_leaves": p["num_leaves"],
        "min_child_samples": p["min_child_samples"], "feature_fraction": p["feature_fraction"],
        "bagging_fraction": p["bagging_fraction"], "bagging_freq": p["bagging_freq"],
        "num_threads": p["num_threads"], "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
        "deterministic": True, "force_col_wise": True,
    }


def train_lgbm(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray,
               cfg: dict[str, Any]) -> lgb.Booster:
    """Unweighted binary LightGBM with early stopping on validation-patient AUPRC.

    No class weighting: probabilities stay calibrated, and AUPRC/AUROC are rank metrics anyway.
    """
    p = cfg["models"]["lgbm"]
    dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, free_raw_data=False)
    return lgb.train(lgbm_params(cfg), dtr, num_boost_round=p["n_estimators"], valid_sets=[dva],
                     callbacks=[lgb.early_stopping(p["early_stopping_rounds"], verbose=False)])


# --------------------------------------------------------------------------- runner
def load_or_build_features(ds: dict[str, Any], cfg: dict[str, Any]) -> pd.DataFrame:
    """Handcrafted features for every window, cached next to the processed dataset."""
    path = resolve_path(cfg["paths"]["processed_dir"]) / "window_features.npz"
    # Fingerprint the inputs so regenerated data can never silently reuse stale features.
    h = hashlib.sha1(ds["series"].tobytes())
    h.update(ds["windows"][["patient_idx", "anchor_idx"]].to_numpy().tobytes())
    h.update(json.dumps([cfg["features"], cfg["windows"], cfg["sampling"]], sort_keys=True).encode())
    fingerprint = h.hexdigest()
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if "fingerprint" in z.files and str(z["fingerprint"]) == fingerprint:
            return pd.DataFrame(z["values"], columns=z["columns"])
    feats = build_feature_table(ds["series"], ds["windows"], ds["timestamps"], cfg)
    np.savez_compressed(path, values=feats.to_numpy(np.float32), columns=np.array(feats.columns).astype(str),
                        fingerprint=np.array(fingerprint))
    return feats


def results_markdown(df: pd.DataFrame, title: str) -> str:
    """Compact results table: AUPRC first (headline for hypo), then AUROC and sensitivity at 10% FAR."""
    def ci(r: pd.Series, m: str) -> str:
        return f"{r[m]:.3f} ({r[m + '_lo']:.3f}-{r[m + '_hi']:.3f})"
    lines = [f"# {title}", "",
             "Test patients only. 95% CIs from a patient-level bootstrap. **SYNTHETIC DATA, not clinically validated.**", "",
             "*onset* = patient not already past the threshold at prediction time (headline); *all* = every window.",
             "FAR = false-alert rate (fraction of negative windows that trigger an alert). Alert burden counts",
             "alerting 30-min windows per monitored patient-day (nocturnal rows: per patient-night), not episodes.", "",
             "| Target | Subset | Model | Prevalence | AUPRC | AUROC | Sens @ 10% FAR | Alerting windows / patient-day @ 10% FAR |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.target} | {r.subset} | {r.model} | {r.prevalence:.3f} | {ci(r, 'auprc')} | {ci(r, 'auroc')} | "
                     f"{r.sens_at_far10:.3f} | {r.alert_windows_per_patient_day_far10:.1f} |")
    return "\n".join(lines) + "\n"


def run(cfg: dict[str, Any], feature_sets: list[str]) -> pd.DataFrame:
    ds = load_processed(cfg)
    windows = ds["windows"]
    feats = load_or_build_features(ds, cfg)
    split = windows.split.to_numpy()
    tr, va, te = (split == s for s in ("train", "val", "test"))
    test_w = windows[te].reset_index(drop=True)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    results_dir = resolve_path(Path(cfg["paths"]["processed_dir"]) / "results")

    horizon_min = cfg["windows"]["horizon_hours"] * 60
    preds = {"window_id": np.flatnonzero(te)}
    rows = []
    pers = persistence_scores(feats[te], horizon_min)
    rows += [{"model": "persistence", **r} for r in evaluate_targets(test_w, pers, cfg)]
    preds.update({f"persistence_{t}": pers[t] for t in TARGETS})

    meta = {}
    for fs in feature_sets:
        X = feature_matrix(feats, windows, ds["static"], fs)
        scores = {}
        for target in TARGETS:
            y = windows[target].to_numpy(int)
            booster = train_lgbm(X[tr], y[tr], X[va], y[va], cfg)
            booster.save_model(str(models_dir / f"lgbm_{fs}_{target}.txt"))
            scores[target] = booster.predict(X[te], num_iteration=booster.best_iteration)
            meta[f"lgbm_{fs}_{target}"] = {"best_iteration": booster.best_iteration, "n_features": X.shape[1]}
            print(f"  lgbm[{fs}] {target}: best_iter={booster.best_iteration}", flush=True)
        rows += [{"model": f"lgbm_{fs}", **r} for r in evaluate_targets(test_w, scores, cfg)]
        preds.update({f"lgbm_{fs}_{t}": scores[t] for t in TARGETS})

    res = pd.DataFrame(rows)
    res.to_csv(results_dir / "baselines_metrics.csv", index=False)
    pd.DataFrame(preds).to_csv(results_dir / "baselines_test_predictions.csv.gz", index=False)
    (models_dir / "lgbm_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sub_order = {"onset": 0, "all": 1, "nocturnal_onset": 2, "nocturnal": 3}
    res = res.sort_values(["target", "subset"], key=lambda c: c.map(sub_order) if c.name == "subset" else c, kind="stable")
    (resolve_path(cfg["paths"]["docs_dir"]) / "baseline_results.md").write_text(
        results_markdown(res, "Stage 3 baseline results"), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate Stage 3 baselines")
    ap.add_argument("--config", default=None)
    ap.add_argument("--feature-sets", default=",".join(FEATURE_SETS), help="comma-separated subset of " + ",".join(FEATURE_SETS))
    args = ap.parse_args()
    res = run(load_config(args.config), args.feature_sets.split(","))
    cols = ["target", "subset", "model", "prevalence", "auprc", "auprc_lo", "auprc_hi", "auroc", "sens_at_far10"]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
