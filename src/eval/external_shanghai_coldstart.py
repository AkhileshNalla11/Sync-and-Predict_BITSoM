"""Cold start on ShanghaiT2DM: does the EHR help when only 0-2 hours of CGM history exist (a new patient)?

    python -m src.eval.external_shanghai_coldstart --data-dir data/external/shanghai

With 6 h of CGM the EHR added nothing (docs/external_shanghai.md). A new patient has little or no CGM yet,
which is where a static record should matter most. For each history length H in HISTORY_HOURS:
  * every window keeps only its last H hours of CGM (earlier readings set to missing) and the CGM features
    are recomputed from that; H = 0 leaves only the time of day;
  * the same windows, labels, patient-grouped folds and paired patient bootstrap as external_shanghai;
  * models: LightGBM on CGM(H) + time vs + our 24 EHR features, and the low-capacity check (logistic
    regression on the CGM(H) model's logit, without vs with 8 core EHR fields).
The window set does not change with H, so rows are comparable across history lengths.
Written to docs/external_shanghai_coldstart.csv (metrics and paired differences, aggregate only).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ROOT, load_config, resolve_path
from src.data.ehr_features import STATIC_FEATURES
from src.data.shanghai import Period, load_cohort
from src.data.wearable_features import SERIES_CHANNELS, extract_windows, handcrafted_features, window_sizes
from src.eval.external_shanghai import CGM, build_windows, cv_scores, stacked_scores
from src.eval.metrics import auprc, auroc, calibration_summary, paired_bootstrap_diff
from src.models.baselines import TARGETS

HISTORY_HOURS = (0.0, 0.5, 1.0, 2.0, 6.0)


def short_history_features(periods: list[Period], windows: pd.DataFrame, hours: float, cfg: dict[str, Any]) -> pd.DataFrame:
    """CGM + time features for every window as if only the last `hours` of CGM had been recorded."""
    history, _, _ = window_sizes(cfg)
    dt = cfg["sampling"]["interval_minutes"]
    keep = int(round(hours * 60 / dt))
    parts = []
    for i, p in enumerate(periods):
        rows = windows.index[windows.patient_idx == i]
        if not len(rows):
            continue
        series = np.full((1, len(p.cgm), len(SERIES_CHANNELS)), np.nan, np.float32)
        series[0, :, 0] = p.cgm.to_numpy(np.float32)
        anchors = windows.anchor_idx.to_numpy()[rows]
        raw = extract_windows(series, np.zeros(len(rows), int), anchors, history).copy()
        raw[:, : history - keep, 0] = np.nan                    # forget everything before the last `hours`
        ts = p.cgm.index
        f = handcrafted_features(raw, anchors, ts[0].hour + ts[0].minute / 60, dt, cfg)
        parts.append(f.set_index(rows))
    return pd.concat(parts).sort_index()[list(CGM)]


def run(cfg: dict[str, Any], data_dir: Path) -> pd.DataFrame:
    periods, static, _ = load_cohort(data_dir, cfg["sampling"]["interval_minutes"])
    w, full = build_windows(periods, static, cfg)
    ehr = full[list(STATIC_FEATURES)]
    print(f"{len(periods)} periods, {w.patient_id.nunique()} patients, {len(w):,} windows", flush=True)
    rows = []
    for h in HISTORY_HOURS:
        cgm = full[list(CGM)] if h == 6.0 else short_history_features(periods, w, h, cfg)
        models: dict[str, dict[str, np.ndarray]] = {
            "lgbm_cgm": {t: cv_scores(w, cgm, t, cfg) for t in TARGETS},
            "lgbm_cgm_ehr": {t: cv_scores(w, pd.concat([cgm, ehr], axis=1), t, cfg) for t in TARGETS},
        }
        feats = pd.concat([cgm, full[[c for c in full.columns if c not in cgm.columns]]], axis=1)
        stacks = {t: stacked_scores(w, feats, models["lgbm_cgm"][t], t, cfg) for t in TARGETS}
        models["stack_cgm"] = {t: v[0] for t, v in stacks.items()}
        models["stack_cgm_ehr"] = {t: v[1] for t, v in stacks.items()}
        for t in TARGETS:
            for subset in ("onset", "nocturnal_onset") if t == "hypo" else ("onset",):
                m = ~w[f"in_event_{t}"].to_numpy(bool)
                if subset == "nocturnal_onset":
                    m &= w.nocturnal.to_numpy(bool)
                y, g = w[t].to_numpy(int)[m], w.patient_id.to_numpy()[m]
                base = {"history_hours": h, "target": t, "subset": subset, "n_patients": int(np.unique(g).size),
                        "n_positive": int(y.sum()), "prevalence": float(y.mean())}
                for name, sc in models.items():
                    rows.append({**base, "row": "model", "model": name, "auprc": auprc(y, sc[t][m]),
                                 "auroc": auroc(y, sc[t][m]), "ece": calibration_summary(y, sc[t][m])["ece"]})
                for a, b in (("lgbm_cgm_ehr", "lgbm_cgm"), ("stack_cgm_ehr", "stack_cgm")):
                    for metric, fn in (("auprc", auprc), ("auroc", auroc)):
                        d = paired_bootstrap_diff(y, models[a][t][m], models[b][t][m], g, fn,
                                                  cfg["eval"]["n_bootstrap"], cfg["seed"])
                        rows.append({**base, "row": "paired", "model": f"{a} - {b}", "metric": metric, **d})
        print(f"  history {h:g} h done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cold-start EHR check on ShanghaiT2DM")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "external" / "shanghai"))
    ap.add_argument("--out-dir", default=None, help="defaults to paths.docs_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out_dir) if args.out_dir else resolve_path(cfg["paths"]["docs_dir"])
    res = run(cfg, Path(args.data_dir))
    res.to_csv(out / "external_shanghai_coldstart.csv", index=False)
    pd.set_option("display.width", 250)
    m = res[res.row == "model"].pivot_table(index=["target", "subset", "history_hours"], columns="model", values="auprc")
    print(m.round(3).to_string())
    p = res[(res.row == "paired") & (res.metric == "auprc")]
    print(p[["history_hours", "target", "subset", "model", "diff", "diff_lo", "diff_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
