"""Head-to-head on real patients: our CGM-only model vs CGM-style trend ("rate of fall") low alarms.

    python -m src.eval.trend_comparison --data-dir data/external/shanghai

Question: CGM makers already sell predictive low alerts that extrapolate the recent trend (Dexcom G7 "Urgent
Low Soon", Medtronic's "alert before low"). What does a learned model add over that?

Everything uses the project's own definitions (src/eval/alert_policy.py): episodes >= 15 min below 70 mg/dL;
"warned >= L min" = an alert between 2 h and L min before onset; PPV = alert followed by a reading < 70 within
2 h; 90-min refractory period; no predictive alert while the CGM already reads < 70.

  A  our real-trained model, predictions every 30 min (the result in docs/real_model.md)
  B  the same model and fold thresholds, checked every 5 min like a CGM
  C  fixed trend alarms, checked every 5 min: fire when the 30-min trend line reaches 70 within H minutes
     (H = 20, the Dexcom idea at 70 instead of 55; H = 60, the Medtronic idea)
  D  trend alarms whose alarm level is tuned ON THE TEST DATA to give row B's alerts per day. This favours
     the trend alarms; it is the fairest matched-burden comparison available.
Paired differences B - D resample the same recording periods for both. The real Dexcom / Medtronic algorithms
are not public, so C and D are versions of the same idea, not the products.
Writes docs/trend_comparison.md, docs/trend_comparison.csv and docs/trend_comparison_paired.csv.
RESEARCH PROTOTYPE, NOT FOR CLINICAL USE.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ROOT, load_config, resolve_path
from src.data.shanghai import load_cohort
from src.eval.alert_policy import Split, bootstrap, evaluate, summarise
from src.eval.external_shanghai import CGM, build_windows, folds
from src.eval.metrics import threshold_at_far
from src.eval.real_model import INNER_VAL_FRACTION, real_split
from src.models.baselines import train_lgbm
from src.twin.alerts import AlertPolicy, policy_far

TARGET = "hypo"
LEVEL = 70.0                      # projected-glucose level of the fixed trend alarms (our hypo threshold)
FIXED = ((20, "Trend alarm: projected to reach 70 within 20 min (the Dexcom idea)"),
         (60, "Trend alarm: projected to reach 70 within 60 min (the Medtronic idea)"))
MATCHED_H = (20, 30, 60)
N_PAIRED = 1000
METRICS = ("caught_15", "caught_30", "ppv", "alerts_per_day")


def model_risk(w30: pd.DataFrame, X30: pd.DataFrame, w5: pd.DataFrame, X5: pd.DataFrame,
               cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold risk divided by its fold's alert threshold (alert when >= 1), at the 30- and 5-min strides.

    Same patient-grouped folds, inner validation patients and threshold rule as src/eval/real_model.py.
    """
    far = policy_far(cfg)[TARGET]
    p30, p5 = w30.patient_id.to_numpy(), w5.patient_id.to_numpy()
    y = w30[TARGET].to_numpy(int)
    s30, s5 = np.full(len(w30), np.nan), np.full(len(w5), np.nan)
    for k, test_pts in enumerate(folds(p30, cfg["seed"])):
        rest = np.setdiff1d(np.unique(p30), test_pts)
        n_val = max(5, round(INNER_VAL_FRACTION * len(rest)))
        val_pts = np.random.default_rng([cfg["seed"], 100 + k]).choice(rest, n_val, replace=False)
        te, va = np.isin(p30, test_pts), np.isin(p30, val_pts)
        tr = ~te & ~va
        b = train_lgbm(X30[tr], y[tr], X30[va], y[va], cfg)
        onset_va = va & ~w30[f"in_event_{TARGET}"].to_numpy(bool)
        th = threshold_at_far(y[onset_va], b.predict(X30[onset_va], num_iteration=b.best_iteration), far)
        te5 = np.isin(p5, test_pts)
        s30[te] = b.predict(X30[te], num_iteration=b.best_iteration) / th
        s5[te5] = b.predict(X5[te5], num_iteration=b.best_iteration) / th
    return s30, s5


class Bench:
    """Runs one risk stream through the configured policy on one stride and returns metrics + per-period counts."""

    def __init__(self, periods: list, w: pd.DataFrame, cfg: dict[str, Any]):
        self.periods, self.w, self.cfg = periods, w, cfg
        self.refractory = float(cfg["twin"]["alert_policy"][TARGET]["refractory_min"])

    def run(self, risk: np.ndarray, threshold: float) -> tuple[dict[str, float], pd.DataFrame, Split]:
        split = real_split(self.periods, self.w, {TARGET: risk, "hyper": np.zeros(len(self.w))}, self.cfg)
        res, pp = evaluate(split, AlertPolicy(threshold, refractory_min=self.refractory, suppress_in_event=True), TARGET)
        return res, pp, split

    def with_ci(self, label: str, risk: np.ndarray, threshold: float) -> tuple[dict[str, Any], pd.DataFrame, Split]:
        res, pp, split = self.run(risk, threshold)
        ci = bootstrap(pp, split, self.cfg["eval"]["n_bootstrap"], self.cfg["seed"])
        row = {"alarm": label, **res, **{f"{k}_lo": v[0] for k, v in ci.items()}, **{f"{k}_hi": v[1] for k, v in ci.items()}}
        return row, pp, split


def trend_risk(feats: pd.DataFrame, horizon_min: float, level: float) -> np.ndarray:
    """>= 0 when the last reading plus the 30-min slope x horizon is at or below `level`."""
    proj = feats.cgm_last.to_numpy(float) + np.nan_to_num(feats.cgm_slope_30m.to_numpy(float), nan=0.0) * horizon_min
    return np.where(np.isfinite(proj), level - proj, -np.inf)


def matched_level(bench: Bench, feats: pd.DataFrame, horizon_min: float, target_rate: float) -> float:
    """Alarm level (mg/dL) giving the target alerts per day; alerts rise with the level, so bisect."""
    lo, hi = 40.0, 140.0
    for _ in range(25):
        mid = (lo + hi) / 2
        rate = bench.run(trend_risk(feats, horizon_min, mid), 0.0)[0]["alerts_per_day"]
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    return (lo + hi) / 2


def paired(pp_a: pd.DataFrame, pp_b: pd.DataFrame, split: Split, seed: int) -> dict[str, float]:
    """A - B with both resampled on the same recording periods."""
    a = pp_a.sort_index()
    b = pp_b.reindex(a.index).fillna(0)
    sa, sb = summarise(a, split), summarise(b, split)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(N_PAIRED):
        idx = rng.integers(0, len(a), len(a))
        ra = summarise(a.iloc[idx].reset_index(drop=True), split)
        rb = summarise(b.iloc[idx].reset_index(drop=True), split)
        diffs.append({m: ra[m] - rb[m] for m in METRICS})
    d = pd.DataFrame(diffs)
    out = {}
    for m in METRICS:
        out[m] = sa[m] - sb[m]
        out[f"{m}_lo"], out[f"{m}_hi"] = float(np.nanpercentile(d[m], 2.5)), float(np.nanpercentile(d[m], 97.5))
        out[f"{m}_p_a_better"] = float((d[m] > 0).mean())
    return out


def pct(r: pd.Series, c: str) -> str:
    return f"{r[c]:.0%} ({r[c + '_lo']:.0%}-{r[c + '_hi']:.0%})"


def pts(r: pd.Series, m: str) -> str:
    return f"{100 * r[m]:+.1f} ({100 * r[m + '_lo']:+.1f} to {100 * r[m + '_hi']:+.1f})"


def write_report(rows: pd.DataFrame, pair: pd.DataFrame, n_periods: int, n_patients: int, path: Path) -> None:
    b = rows[rows.alarm.str.startswith("B.")].iloc[0]
    a = rows[rows.alarm.str.startswith("A.")].iloc[0]
    best = pair.iloc[0]
    lines = [
        "# Our model vs CGM-style trend alarms (real patients)", "",
        "**Research prototype. Not a medical device, not for clinical use.** Generated by",
        "`python -m src.eval.trend_comparison --data-dir data/external/shanghai` (ShanghaiT2DM, CC BY 4.0, not redistributed).", "",
        "CGM makers already sell predictive low alerts that extrapolate the recent trend (Dexcom G7 \"Urgent Low Soon\",",
        ("Medtronic's \"alert before low\"). This page asks what a learned model adds over that idea, on "
         f"{n_patients} real patients ({n_periods} recording periods, {int(b.episodes)} low episodes)."), "",
        "## Result", "",
        (f"**At the same number of alerts (about {b.alerts_per_day:.1f} a day), our model's alerts were followed by a real low "
         f"more often ({b.ppv:.0%} vs {best.trend_ppv:.0%} for a 20-minute trend alarm) and warned of {b.caught_30:.0%} of lows "
         f"at least 30 minutes ahead (vs {best.trend_caught_30:.0%}), with a median warning of {b.median_lead_min:.0f} min. "
         "The edge is real but modest: a well-tuned trend alarm is just as good at 15 minutes.**"), "",
        "| Alarm | Warned >= 15 min | Warned >= 30 min | Median lead (min) | Alerts / patient-day | PPV per alert |",
        "|---|---|---|---|---|---|",
        *[f"| {r.alarm} | {r.caught_15:.0%} | {pct(r, 'caught_30')} | {r.median_lead_min:.0f} | "
          f"{r.alerts_per_day:.2f} ({r.alerts_per_day_lo:.2f}-{r.alerts_per_day_hi:.2f}) | {pct(r, 'ppv')} |"
          for _, r in rows.iterrows()], "",
        "95% CIs resample recording periods.", "",
        "## Paired differences: our model (B) minus each matched trend alarm (D)", "",
        f"Percentage points, {N_PAIRED} paired resamples of recording periods. *P* = share of resamples where our model is better.", "",
        "| Trend alarm | Warned >= 15 min | Warned >= 30 min | PPV | Alerts / day | P (>= 30 min) | P (PPV) |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r.trend} | {pts(r, 'caught_15')} | {pts(r, 'caught_30')} | {pts(r, 'ppv')} | "
          f"{r.alerts_per_day:+.2f} ({r.alerts_per_day_lo:+.2f} to {r.alerts_per_day_hi:+.2f}) | "
          f"{r.caught_30_p_a_better:.2f} | {r.ppv_p_a_better:.2f} |" for _, r in pair.iterrows()], "",
        "## Reading this honestly", "",
        "* **The prediction adds a real but modest gain over a tuned trend alarm.** More of its alerts are real and the",
        "  warning comes earlier; at 15 minutes ahead a simple trend alarm is as good. The value of the product therefore",
        "  rests as much on the reasons, the record context and the clinician view as on the forecast itself.",
        (f"* **Checking every 5 minutes matters.** The same model checked every 5 minutes instead of every 30 warns of "
         f"{b.caught_30:.0%} instead of {a.caught_30:.0%} of lows at least 30 minutes ahead ({b.caught_15:.0%} vs "
         f"{a.caught_15:.0%} at 15 min), at {b.alerts_per_day:.2f} instead of {a.alerts_per_day:.2f} alerts a day. "
         "The deployed alerts should move to 5-minute checks, with the alert budget re-tuned with clinicians."),
        "* **A long-horizon trend alarm buys early warnings with many false alarms.** Projecting 60 minutes ahead catches",
        "  most lows early but needs far more alerts, most of them false.", "",
        "## Caveats", "",
        "* The trend alarms' level was tuned on the test data itself (row D); our thresholds were set on held-out training",
        "  patients. This favours the trend alarms.",
        "* The real Dexcom and Medtronic algorithms are not public; rows C and D implement the trend idea at our",
        "  episode threshold (70 mg/dL; Dexcom's alert targets 55).",
        "* 100 patients from one Chinese centre, partly hospitalised; 15-min CGM interpolated to 5 min, which smooths",
        "  the trend. Not Indian outpatients.",
        "* The 30-minute warning difference against the best trend alarm is not yet statistically clear; a larger",
        "  cohort is needed.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "external" / "shanghai"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    docs = resolve_path(cfg["paths"]["docs_dir"])

    periods, static, _ = load_cohort(Path(args.data_dir), cfg["sampling"]["interval_minutes"])
    w30, f30 = build_windows(periods, static, cfg)
    cfg5 = copy.deepcopy(cfg)
    cfg5["windows"]["stride_minutes"] = cfg["sampling"]["interval_minutes"]
    w5, f5 = build_windows(periods, static, cfg5)
    print(f"{len(periods)} periods; {len(w30):,} windows at 30 min, {len(w5):,} at 5 min", flush=True)

    s30, s5 = model_risk(w30, f30[list(CGM)], w5, f5[list(CGM)], cfg)
    bench30, bench5 = Bench(periods, w30, cfg), Bench(periods, w5, cfg5)
    row_a, _, _ = bench30.with_ci("A. Our model, checked every 30 min (deployed)", s30, 1.0)
    row_b, pp_b, split5 = bench5.with_ci("B. Our model, checked every 5 min", s5, 1.0)
    rows = [row_a, row_b]
    for h, label in FIXED:
        rows.append(bench5.with_ci(f"C. {label}", trend_risk(f5, h, LEVEL), 0.0)[0])
    pair_rows = []
    for h in MATCHED_H:
        level = matched_level(bench5, f5, h, row_b["alerts_per_day"])
        row, pp, _ = bench5.with_ci(f"D. Trend alarm, {h}-min projection, level {level:.0f} mg/dL (matched alerts/day)",
                                    trend_risk(f5, h, level), 0.0)
        rows.append(row)
        pair_rows.append({"trend": f"{h}-min projection, level {level:.0f} mg/dL",
                          "trend_ppv": row["ppv"], "trend_caught_30": row["caught_30"],
                          **paired(pp_b, pp, split5, cfg["seed"])})
        print(rows[-1]["alarm"], {m: round(row[m], 3) for m in METRICS}, flush=True)

    out = pd.DataFrame(rows)
    pair = pd.DataFrame(pair_rows)          # 20-min projection first: the CGM-style horizon, our headline comparison
    keep = ["alarm", "episodes", "caught_15", "caught_30", "median_lead_min", "alerts_per_day", "ppv",
            *[f"{m}_{s}" for m in ("caught_30", "alerts_per_day", "ppv") for s in ("lo", "hi")]]
    out[keep].round(4).to_csv(docs / "trend_comparison.csv", index=False)
    pair.round(4).to_csv(docs / "trend_comparison_paired.csv", index=False)
    write_report(out, pair, len(periods), int(w30.patient_id.nunique()), docs / "trend_comparison.md")
    print(out[["alarm", "caught_15", "caught_30", "median_lead_min", "alerts_per_day", "ppv"]].round(3).to_string(index=False))
    print(f"report -> {docs / 'trend_comparison.md'}")


if __name__ == "__main__":
    main()
