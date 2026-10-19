"""Evidence-backed model: a CGM-only LightGBM trained on real patients (ShanghaiT2DM), with its alert policy
evaluated episode by episode on real patients, saved for the dashboard.

    python -m src.eval.real_model --data-dir data/external/shanghai

The real linked-data check (docs/external_shanghai.md) showed that on real patients a CGM-only model trained on
real data beats both the synthetic-trained model and every EHR variant. This script turns that evidence into
the product:
  1) Patient-grouped 10-fold cross-validation on 100 real patients (folds shared with external_shanghai).
     In each fold, training patients are split again: most train the model, the rest set early stopping AND
     the alert thresholds (the policy's false-alert rate on their onset windows, as on synthetic data). The
     policy (config twin.alert_policy: one alert per episode, 90-min refractory, none while the CGM is already
     past the threshold) is then applied to the held-out patients, so no threshold ever sees its test patients.
  2) For comparison, the synthetic-trained CGM-only model with thresholds chosen on SYNTHETIC validation
     patients, applied unchanged ("does the synthetic alerting transfer?").
  3) Episode-level metrics with the alert_policy definitions: episodes (>= 15 min past the threshold) warned
     >= 15 / 30 min ahead, alerts per day, PPV per alert, median lead. 95% CIs resample recording periods.
  4) The deployable model: trained on all 100 patients for the median number of boosting rounds found in
     cross-validation; thresholds at the same false-alert rate on the out-of-fold scores. Saved as
     <models_dir>/lgbm_cgm_shanghai_{hyper,hypo}.txt + _alert_thresholds.json (model files hold tree splits
     only, no patient data; ShanghaiT2DM is CC BY 4.0).
  5) Reverse check: the real-trained model on the 60 SYNTHETIC test patients the dashboard shows.
Writes docs/real_model.md and docs/real_model_alerts.csv. Aggregate numbers only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ROOT, load_config, resolve_path
from src.data.build_dataset import load_processed
from src.data.shanghai import load_cohort
from src.data.wearable_features import is_nocturnal
from src.eval.alert_policy import EPISODE_MIN, LEADS_MIN, Split, bootstrap, evaluate
from src.eval.external_shanghai import CGM, build_windows, folds
from src.eval.metrics import auprc, auroc, event_episodes, threshold_at_far
from src.models.baselines import feature_matrix, lgbm_params, load_or_build_features, train_lgbm
from src.twin.alerts import policy_far, policy_from_spec

MODEL_NAME = "lgbm_cgm_shanghai"
INNER_VAL_FRACTION = 0.2
TARGETS = ("hyper", "hypo")


def real_split(periods, w: pd.DataFrame, risk: dict[str, np.ndarray], cfg: dict[str, Any]) -> Split:
    """alert_policy.Split for ShanghaiT2DM: one 'patient' per recording period (each has its own time grid)."""
    dt = cfg["sampling"]["interval_minutes"]
    order = np.lexsort((w.anchor_idx.to_numpy(), w.patient_idx.to_numpy()))
    w = w.iloc[order].reset_index(drop=True).assign(patient_id=lambda d: d.period_id)
    risk = {t: np.asarray(v)[order] for t, v in risk.items()}
    thr = cfg["thresholds"]
    n = EPISODE_MIN // dt
    eps = {}
    for t, level, below in (("hypo", thr["hypo_mgdl"], True), ("hyper", thr["hyper_mgdl"], False)):
        rows = []
        for i in np.unique(w.patient_idx):
            p = periods[i]
            ons = event_episodes(p.cgm.to_numpy(float)[None], level, below=below, min_samples=n, clear_samples=n)[0]
            noct = is_nocturnal(p.cgm.index[ons], thr["nocturnal_start"], thr["nocturnal_end"]) if len(ons) else []
            rows += [(i, int(o), bool(nn)) for o, nn in zip(ons, noct)]
        eps[t] = pd.DataFrame(rows, columns=["patient_idx", "onset_idx", "nocturnal"])
    horizon = cfg["windows"]["horizon_hours"] * 60 // dt
    return Split(w, risk, eps, cfg["windows"]["stride_minutes"] / 60, horizon, dt)


def cv_with_thresholds(w: pd.DataFrame, X: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict, dict, dict]:
    """Out-of-fold risk, per-window thresholds (each set on its fold's inner validation patients), best iterations."""
    far = policy_far(cfg)
    pids = w.patient_id.to_numpy()
    risk = {t: np.full(len(w), np.nan) for t in TARGETS}
    thr = {t: np.full(len(w), np.nan) for t in TARGETS}
    iters: dict[str, list[int]] = {t: [] for t in TARGETS}
    for k, test_pts in enumerate(folds(pids, cfg["seed"])):
        rest = np.setdiff1d(np.unique(pids), test_pts)
        n_val = max(5, int(round(INNER_VAL_FRACTION * len(rest))))
        val_pts = np.random.default_rng([cfg["seed"], 100 + k]).choice(rest, n_val, replace=False)
        te, va = np.isin(pids, test_pts), np.isin(pids, val_pts)
        tr = ~te & ~va
        for t in TARGETS:
            y = w[t].to_numpy(int)
            b = train_lgbm(X[tr], y[tr], X[va], y[va], cfg)
            iters[t].append(int(b.best_iteration))
            onset_va = va & ~w[f"in_event_{t}"].to_numpy(bool)
            p_va = b.predict(X[onset_va], num_iteration=b.best_iteration)
            thr[t][te] = threshold_at_far(y[onset_va], p_va, far[t])
            risk[t][te] = b.predict(X[te], num_iteration=b.best_iteration)
    return risk, thr, iters


def policy_metrics(split: Split, thresholds: dict[str, np.ndarray | float], cfg: dict[str, Any], label: str) -> list[dict]:
    """Episode metrics with the configured policy. Per-window thresholds are handled by rescaling the risk."""
    rows = []
    for t in TARGETS:
        th = np.broadcast_to(np.asarray(thresholds[t], float), split.risk[t].shape)
        # policy compares risk >= threshold; dividing by a per-window threshold gives one common threshold of 1.0
        scaled = Split(split.windows, {**split.risk, t: split.risk[t] / th}, split.episodes, split.stride_h,
                       split.horizon, split.dt)
        pol = policy_from_spec(cfg["twin"]["alert_policy"][t], 1.0)
        res, pp = evaluate(scaled, pol, t)
        ci = bootstrap(pp, scaled, cfg["eval"]["n_bootstrap"], cfg["seed"])
        rows.append({"model": label, "target": t, **res, **{f"{k}_lo": v[0] for k, v in ci.items()},
                     **{f"{k}_hi": v[1] for k, v in ci.items()}})
    return rows


def synthetic_thresholds(cfg: dict[str, Any], models_dir: Path) -> dict[str, float]:
    """The synthetic-trained CGM-only model's thresholds at the policy's false-alert rate on SYNTHETIC validation."""
    ds = load_processed(cfg)
    feats = load_or_build_features(ds, cfg)
    rows = (ds["windows"].split == "val").to_numpy()
    w = ds["windows"][rows]
    X = feature_matrix(feats[rows], w, ds["static"], "cgm")
    far = policy_far(cfg)
    out = {}
    for t in TARGETS:
        b = lgb.Booster(model_file=str(models_dir / f"lgbm_cgm_{t}.txt"))
        onset = ~w[f"in_event_{t}"].to_numpy(bool)
        out[t] = threshold_at_far(w[t].to_numpy(int)[onset], b.predict(X[onset]), far[t])
    return out


def reverse_check(cfg: dict[str, Any], boosters: dict[str, lgb.Booster], models_dir: Path) -> pd.DataFrame:
    """Real-trained vs synthetic-trained models on the SYNTHETIC test patients (onset windows)."""
    ds = load_processed(cfg)
    feats = load_or_build_features(ds, cfg)
    rows = (ds["windows"].split == "test").to_numpy()
    w = ds["windows"][rows]
    out = []
    for name, feature_set, get in (("Real-trained, CGM only (ShanghaiT2DM)", "cgm", lambda t: boosters[t]),
                                   ("Synthetic-trained, CGM only", "cgm",
                                    lambda t: lgb.Booster(model_file=str(models_dir / f"lgbm_cgm_{t}.txt"))),
                                   ("Synthetic-trained, CGM + wearables + EHR (deployed)", "cgm_wearables_ehr",
                                    lambda t: lgb.Booster(model_file=str(models_dir / f"lgbm_cgm_wearables_ehr_{t}.txt")))):
        X = feature_matrix(feats[rows], w, ds["static"], feature_set)
        for t in TARGETS:
            onset = ~w[f"in_event_{t}"].to_numpy(bool)
            y, s = w[t].to_numpy(int)[onset], get(t).predict(X[onset])
            out.append({"model": name, "target": t, "auroc": auroc(y, s), "auprc": auprc(y, s), "prevalence": y.mean()})
    return pd.DataFrame(out)


def fmt_pct(r: pd.Series, c: str) -> str:
    return f"{r[c]:.0%} ({r[c + '_lo']:.0%}-{r[c + '_hi']:.0%})"


def write_report(alerts: pd.DataFrame, rev: pd.DataFrame, saved: dict[str, Any], n_periods: int, n_patients: int,
                 path: Path) -> None:
    def row(label: str, t: str) -> str:
        r = alerts[(alerts.model == label) & (alerts.target == t)].iloc[0]
        return (f"| {label} | {fmt_pct(r, 'caught_15')} | {fmt_pct(r, 'caught_30')} | {r.median_lead_min:.0f} | "
                f"{r.alerts_per_day:.2f} ({r.alerts_per_day_lo:.2f}-{r.alerts_per_day_hi:.2f}) | {fmt_pct(r, 'ppv')} |")
    real, syn = "Real-trained, CGM only", "Synthetic-trained, CGM only, zero-shot"
    h = alerts[(alerts.model == real) & (alerts.target == "hypo")].iloc[0]
    lines = [
        "# Evidence-backed model: trained and alert-tested on real patients", "",
        "**Research prototype. Not a medical device, not for clinical use.** Generated by",
        "`python -m src.eval.real_model --data-dir data/external/shanghai` (ShanghaiT2DM, CC BY 4.0, not redistributed).", "",
        "The real linked-data check ([external_shanghai.md](external_shanghai.md)) found that on real patients a",
        "CGM-only model trained on real data beats the synthetic-trained model and every EHR variant. This page",
        "trains that model, tests the **alert policy on real patients**, and saves it for the dashboard, where it can be",
        "selected as the risk model (the default stays the synthetic-trained model, which matches the synthetic",
        "patients the dashboard shows).", "",
        "## Alert policy on real patients", "",
        f"{n_patients} real patients ({n_periods} recording periods), patient-grouped 10-fold cross-validation. In each",
        "fold the alert thresholds are set on held-out *training* patients at the policy's false-alert rate (hypo 1%,",
        "hyper 3% of onset windows), then the policy (one alert per episode, 90-min refractory period, none while the",
        "CGM already reads past the threshold) runs on the test patients. Episodes are >= 15 min past the threshold.",
        "95% CIs resample recording periods.", "",
        f"**Hypoglycaemia: {h.caught_30:.0%} of real low-glucose episodes warned at least 30 minutes ahead "
        f"({h.caught_15:.0%} at 15 min), {h.alerts_per_day:.2f} alerts per patient-day, {h.ppv:.0%} of alerts followed "
        f"by a real low, median warning {h.median_lead_min:.0f} min.**", "",
        "| Model | Episodes warned >= 15 min | >= 30 min | Median lead (min) | Alerts / patient-day | PPV per alert |",
        "|---|---|---|---|---|---|",
        "| **Hypoglycaemia (< 70 mg/dL)** | | | | | |", row(real, "hypo"), row(syn, "hypo"),
        "| **Hyperglycaemia (> 180 mg/dL)** | | | | | |", row(real, "hyper"), row(syn, "hyper"), "",
        "* The synthetic-trained row uses thresholds chosen on synthetic validation patients, applied unchanged.",
        "* Episodes: hypo " + f"{int(h.episodes)}" + ", hyper "
        + f"{int(alerts[(alerts.model == real) & (alerts.target == 'hyper')].episodes.iloc[0])}" + ".",
        "* For reference, on synthetic test patients the deployed synthetic model warned 76% of hypo episodes >= 30 min",
        "  ahead at 0.49 alerts per patient-day ([alert_policy.md](alert_policy.md)).", "",
        "## The saved model", "",
        f"* `{MODEL_NAME}_hyper.txt` / `_hypo.txt`: LightGBM on the 24 CGM + time-of-day features, trained on all",
        f"  {n_patients} patients for the median cross-validated number of rounds (hyper {saved['rounds']['hyper']},"
        f" hypo {saved['rounds']['hypo']}).",
        f"* Alert thresholds (`{MODEL_NAME}_alert_thresholds.json`): hyper {saved['hyper']:.3f}, hypo {saved['hypo']:.3f},"
        " at the same false-alert rates on the out-of-fold scores.",
        "* The files hold tree splits only, no patient records.", "",
        "## On the synthetic patients the dashboard shows", "",
        "The dashboard's patients are synthetic, so we also checked the real-trained model on the 60 synthetic test",
        "patients (onset windows). It was never trained on synthetic data, so this is transfer in the other direction.", "",
        "| Model | Hyper AUROC | Hyper AUPRC | Hypo AUROC | Hypo AUPRC |", "|---|---|---|---|---|",
        *[f"| {m} | {g[g.target == 'hyper'].auroc.iloc[0]:.3f} | {g[g.target == 'hyper'].auprc.iloc[0]:.3f} | "
          f"{g[g.target == 'hypo'].auroc.iloc[0]:.3f} | {g[g.target == 'hypo'].auprc.iloc[0]:.3f} |"
          for m, g in rev.groupby("model", sort=False)], "",
        "## Caveats", "",
        "* 100 patients from one Chinese centre, partly hospitalised; 15-min CGM interpolated to 5 min. Not Indian",
        "  outpatients. Real-world alert numbers would need re-tuning with clinicians on the target population.",
        "* The model sees only CGM and time of day. The EHR is still shown in the dashboard as context and used by the",
        "  what-if twin, but it is not an input to this model, because on real data it did not improve the forecast.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "external" / "shanghai"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    docs = resolve_path(cfg["paths"]["docs_dir"])

    periods, static, _ = load_cohort(Path(args.data_dir), cfg["sampling"]["interval_minutes"])
    w, feats = build_windows(periods, static, cfg)
    X = feats[list(CGM)]
    print(f"{len(periods)} periods, {w.patient_id.nunique()} patients, {len(w):,} windows", flush=True)

    risk, thr, iters = cv_with_thresholds(w, X, cfg)
    split = real_split(periods, w, risk, cfg)
    order = np.lexsort((w.anchor_idx.to_numpy(), w.patient_idx.to_numpy()))
    rows = policy_metrics(split, {t: thr[t][order] for t in TARGETS}, cfg, "Real-trained, CGM only")
    syn_thr = synthetic_thresholds(cfg, models_dir)
    syn_risk = {t: lgb.Booster(model_file=str(models_dir / f"lgbm_cgm_{t}.txt")).predict(X) for t in TARGETS}
    rows += policy_metrics(real_split(periods, w, syn_risk, cfg), syn_thr, cfg, "Synthetic-trained, CGM only, zero-shot")
    alerts = pd.DataFrame(rows)
    print(alerts[["model", "target", "episodes", "caught_15", "caught_30", "median_lead_min", "alerts_per_day", "ppv"]]
          .round(3).to_string(index=False), flush=True)

    # deployable model: all patients, median CV rounds; thresholds from the out-of-fold scores
    far = policy_far(cfg)
    boosters, saved = {}, {"rounds": {}}
    for t in TARGETS:
        y = w[t].to_numpy(int)
        rounds = int(np.median(iters[t]))
        boosters[t] = lgb.train(lgbm_params(cfg), lgb.Dataset(X, label=y), num_boost_round=rounds)
        boosters[t].save_model(str(models_dir / f"{MODEL_NAME}_{t}.txt"))
        onset = ~w[f"in_event_{t}"].to_numpy(bool)
        saved[t] = threshold_at_far(y[onset], risk[t][onset], far[t])
        saved["rounds"][t] = rounds
    real_rows = alerts[alerts.model == "Real-trained, CGM only"].set_index("target")
    saved.update(false_alert_rate=far, fitted_on="ShanghaiT2DM", n_patients=int(w.patient_id.nunique()),
                 n_windows=int(len(w)), features="cgm",
                 real_patient_alerts={t: {k: round(float(real_rows.loc[t, k]), 4) for k in
                                          ("caught_15", "caught_30", "alerts_per_day", "ppv", "median_lead_min")}
                                      for t in TARGETS})
    (models_dir / f"{MODEL_NAME}_alert_thresholds.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")

    rev = reverse_check(cfg, boosters, models_dir)
    print(rev.round(3).to_string(index=False))
    alerts.round(4).to_csv(docs / "real_model_alerts.csv", index=False)
    rev.round(4).to_csv(docs / "real_model_synthetic_check.csv", index=False)
    write_report(alerts, rev, saved, len(periods), int(w.patient_id.nunique()), docs / "real_model.md")
    print(f"saved {MODEL_NAME} -> {models_dir}; report -> {docs / 'real_model.md'}")


if __name__ == "__main__":
    main()
