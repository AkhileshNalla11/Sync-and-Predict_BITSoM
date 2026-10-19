"""Calibrated uncertainty bands for the personalised twin's 2-hour glucose projection.

    python -m src.eval.twin_intervals             # after src.eval.twin_validation (reuses / extends the fit cache)

The what-if panel draws the twin's projection for the next 2 h. This module says how far off that line
usually is, using split-conformal prediction intervals:
  1) calibration, VALIDATION patients: on each patient's held-out days (after the `fit_days` used to fit
     the twin), project 2 h ahead every hour exactly as the dashboard does (observer on up to 'now', then
     free-running) and record CGM minus projection at every lead time (5-120 min);
  2) the band at each lead time and starting-glucose level is the empirical quantile range of those
     errors (finite-sample conformal correction), for 80% and 95% coverage;
  3) evaluation, TEST patients: the share of real CGM readings that fall inside the band, overall, by
     lead time and by starting glucose, with patient-bootstrap CIs.
The bands cover the projection "as logged". A what-if scenario changes the inputs, and errors in how the
twin responds to that change are NOT covered: the dashboard says so.
Writes <models_dir>/twin_intervals.json (read by the dashboard) and docs/twin_intervals.{md,csv}.
SYNTHETIC DATA: the twin has the simulator's own structure, so real-world bands would be wider.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.twin import personalise as P

LEVEL_EDGES = (100.0, 180.0)                   # starting-glucose bins: < 100, 100-180, > 180 mg/dL
LEVEL_NAMES = ("< 100", "100-180", "> 180")
COVERAGES = (0.8, 0.95)
ANCHOR_STRIDE_MIN = 60
REPORT_LEADS = (30, 60, 120)
FILE = "twin_intervals.json"


def level_bin(g: float) -> int:
    return int(np.searchsorted(LEVEL_EDGES, g, side="right"))


def patient_residuals(cfg: dict[str, Any], pid: str) -> pd.DataFrame:
    """CGM minus the twin's 2 h projection on held-out days: one row per (anchor, lead) with a reading."""
    dt = cfg["sampling"]["interval_minutes"]
    cal = cfg["simulation"]["calibration"]
    warm = cfg["simulation"]["warmup_days"] * 24 * 60 // dt
    per_day = 24 * 60 // dt
    horizon = cfg["windows"]["horizon_hours"] * 60 // dt
    inp = P.patient_inputs(cfg, pid)
    st = P.settings(cfg, len(inp.series) // per_day)
    fit = P.get_fit(cfg, pid)
    s = P.schedule_from_log(inp, dt, warm)
    cgm = inp.series[:, 0]
    lo = st["fit_days"] * per_day
    anchors = np.arange(lo + per_day // 24, len(cgm) - horizon, ANCHOR_STRIDE_MIN // dt)
    anchors = anchors[~np.isnan(cgm[anchors])]                 # a projection needs a reading at 'now'
    if not len(anchors):
        return pd.DataFrame(columns=["patient_id", "anchor", "lead_min", "level", "resid"])
    n_steps = warm + int(anchors.max()) + horizon + 1
    obs = np.stack([P.observer_target(cgm, warm, n_steps, int(a), dt) for a in anchors])
    rows = fit.params.loc[fit.params.index.repeat(len(anchors))].reset_index(drop=True)
    g = P.simulate_schedule(rows, s, dt, cal, n_steps, obs=obs, obs_gain=st["observer_gain"])[:, warm:]
    fc = np.stack([g[j, a + 1: a + 1 + horizon] for j, a in enumerate(anchors)])        # what the dashboard plots
    truth = np.stack([cgm[a + 1: a + 1 + horizon] for a in anchors])
    lead = np.broadcast_to(np.arange(1, horizon + 1) * dt, truth.shape)
    level = np.broadcast_to(np.array([level_bin(cgm[a]) for a in anchors])[:, None], truth.shape)
    anc = np.broadcast_to(anchors[:, None], truth.shape)
    m = ~np.isnan(truth)
    return pd.DataFrame({"patient_id": pid, "anchor": anc[m], "lead_min": lead[m], "level": level[m],
                         "resid": (truth - fc)[m]})


def split_residuals(cfg: dict[str, Any], split: str, limit: int | None) -> pd.DataFrame:
    proc = resolve_path(cfg["paths"]["processed_dir"])
    pids = sorted(pd.read_csv(proc / "splits.csv").query("split == @split").patient_id)[:limit]
    out = []
    for i, pid in enumerate(pids):
        t0 = time.time()
        out.append(patient_residuals(cfg, pid))
        print(f"[{split} {i + 1}/{len(pids)}] {pid} {len(out[-1]):,} residuals in {time.time() - t0:.1f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def conformal_bands(cal: pd.DataFrame) -> dict[str, Any]:
    """Per coverage, level and lead: (lower, upper) error quantiles with the finite-sample correction."""
    leads = sorted(cal.lead_min.unique())
    bands: dict[str, Any] = {}
    for cov in COVERAGES:
        a = (1 - cov) / 2
        per_level = {}
        for k, name in enumerate(LEVEL_NAMES):
            lo, hi, n_min = [], [], []
            for ld in leads:
                r = cal.resid[(cal.level == k) & (cal.lead_min == ld)].to_numpy()
                if len(r) < 50:                                  # too few: fall back to all levels at this lead
                    r = cal.resid[cal.lead_min == ld].to_numpy()
                n = len(r)
                lo.append(float(np.quantile(r, max(0.0, a * (n + 1) / n - 1 / n), method="lower")))
                hi.append(float(np.quantile(r, min(1.0, (1 - a) * (n + 1) / n), method="higher")))
                n_min.append(n)
            per_level[name] = {"lower": lo, "upper": hi, "n_min": int(min(n_min))}
        bands[f"{cov:g}"] = per_level
    return {"lead_minutes": [int(x) for x in leads], "bands": bands}


def band_for(intervals: dict[str, Any], coverage: float, level_mgdl: float) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) offsets in mg/dL per lead time (5..120 min) for a projection starting at `level_mgdl`."""
    b = intervals["bands"][f"{coverage:g}"][LEVEL_NAMES[level_bin(level_mgdl)]]
    return np.asarray(b["lower"]), np.asarray(b["upper"])


def inside(df: pd.DataFrame, intervals: dict[str, Any], coverage: float) -> np.ndarray:
    leads = np.asarray(intervals["lead_minutes"])
    li = np.searchsorted(leads, df.lead_min.to_numpy())
    lo = np.zeros(len(df))
    hi = np.zeros(len(df))
    for k, name in enumerate(LEVEL_NAMES):
        m = df.level.to_numpy() == k
        b = intervals["bands"][f"{coverage:g}"][name]
        lo[m] = np.asarray(b["lower"])[li[m]]
        hi[m] = np.asarray(b["upper"])[li[m]]
    r = df.resid.to_numpy()
    return (r >= lo) & (r <= hi)


def _boot(per_patient: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float]:
    """Patient-bootstrap CI of pooled coverage (hits / n)."""
    rng = np.random.default_rng(seed)
    h, n = per_patient.hits.to_numpy(), per_patient.n.to_numpy()
    idx = rng.integers(0, len(h), size=(n_boot, len(h)))
    est = h[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


def evaluate(test: pd.DataFrame, intervals: dict[str, Any], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for cov in COVERAGES:
        hit = inside(test, intervals, cov)
        t = test.assign(hit=hit)
        pp = t.groupby("patient_id").hit.agg(hits="sum", n="size").reset_index()
        lo, hi = _boot(pp, n_boot, seed)
        width = []
        for k, name in enumerate(LEVEL_NAMES):
            b = intervals["bands"][f"{cov:g}"][name]
            width.append(np.asarray(b["upper"]) - np.asarray(b["lower"]))
        rows.append({"coverage_target": cov, "group": "all", "coverage": hit.mean(), "ci_lo": lo, "ci_hi": hi,
                     "n": len(t), "mean_width_mgdl": float(np.mean(width))})
        for ld in REPORT_LEADS:
            m = t.lead_min == ld
            rows.append({"coverage_target": cov, "group": f"lead {ld} min", "coverage": t.hit[m].mean(), "n": int(m.sum()),
                         "mean_width_mgdl": float(np.mean([w[intervals["lead_minutes"].index(ld)] for w in width]))})
        for k, name in enumerate(LEVEL_NAMES):
            m = t.level == k
            w = width[k]
            rows.append({"coverage_target": cov, "group": f"start {name} mg/dL", "coverage": t.hit[m].mean(),
                         "n": int(m.sum()), "mean_width_mgdl": float(np.mean(w))})
    return pd.DataFrame(rows)


def write_report(res: pd.DataFrame, intervals: dict[str, Any], n_cal: int, n_test: int, path) -> None:
    a80 = res[(res.coverage_target == 0.8) & (res.group == "all")].iloc[0]
    a95 = res[(res.coverage_target == 0.95) & (res.group == "all")].iloc[0]
    b = intervals["bands"]["0.8"]
    li = {ld: intervals["lead_minutes"].index(ld) for ld in REPORT_LEADS}
    lines = [
        "# Uncertainty bands for the twin's 2-hour projection", "",
        "**SYNTHETIC DATA. Not clinically validated.** Generated by `python -m src.eval.twin_intervals`.", "",
        "The what-if panel draws the personalised twin's glucose projection for the next 2 hours. These bands say",
        "how far off that line usually is. They are **split-conformal prediction intervals**: calibrated on",
        f"{n_cal} validation patients' held-out days, then checked on {n_test} test patients the bands never saw.", "",
        "## Result", "",
        f"* **80% band:** real CGM fell inside it **{a80.coverage:.1%}** of the time on test patients "
        f"(95% CI {a80.ci_lo:.1%}-{a80.ci_hi:.1%}, {int(a80.n):,} readings).",
        f"* **95% band:** **{a95.coverage:.1%}** (95% CI {a95.ci_lo:.1%}-{a95.ci_hi:.1%}).",
        "* 80% band, starting glucose 100-180 mg/dL: "
        + ", ".join(f"{ld} min ahead {b['100-180']['lower'][i]:+.0f} to {b['100-180']['upper'][i]:+.0f} mg/dL"
                    for ld, i in li.items()) + ". The bands are narrow because the twin has the simulator's own",
        "  structure and an exact meal / dose log; on real patients they would be much wider.",
        "* Coverage is close to target overall but uneven by starting glucose: "
        + ", ".join(f"{r.group.removeprefix('start ')} {r.coverage:.0%}"
                    for r in res[(res.coverage_target == 0.8) & res.group.str.startswith("start")].itertuples())
        + " for the 80% band (the bands are set per level on validation patients; above 180 mg/dL test patients miss a little more often).",
        "",
        "| Band | Group | Coverage on test | Readings | Mean width (mg/dL) |", "|---|---|---|---|---|",
        *[f"| {r.coverage_target:.0%} | {r.group} | {r.coverage:.1%} | {int(r.n):,} | {r.mean_width_mgdl:.0f} |"
          for r in res.itertuples()],
        "",
        "## Method", "",
        "1. For each validation patient, the twin is fitted on the first week (as in the dashboard) and, on every",
        f"   held-out hour, projects 2 h ahead: the state observer follows the CGM up to 'now', then the model runs",
        "   freely on the logged meals and doses. The error is CGM minus projection at each 5-min lead time.",
        "2. The band at each lead time and starting-glucose level (< 100, 100-180, > 180 mg/dL) is the range between",
        "   the lower and upper error quantiles, with the finite-sample conformal correction.",
        "3. Coverage is the share of test patients' real readings inside the band; CIs resample whole patients.", "",
        "## Limits", "",
        "* Coverage is marginal (averaged over readings), not guaranteed per patient or per moment.",
        "* Readings from the same projection are correlated, so the CIs are by patient, not by reading.",
        "* The bands cover the projection **as logged**. A what-if scenario changes the inputs; errors in how the",
        "  twin responds to that change are not covered. The dashboard draws the band and says this.",
        "* The twin has the simulator's own structure and the log is exact here, so real-world bands would be wider.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N patients per split (quick check)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    cal = split_residuals(cfg, "val", args.limit)
    test = split_residuals(cfg, "test", args.limit)
    intervals = conformal_bands(cal)
    res = evaluate(test, intervals, cfg["eval"]["n_bootstrap"], cfg["seed"])
    intervals.update(calibrated_on="validation patients, held-out days", n_calibration_patients=int(cal.patient_id.nunique()),
                     level_edges_mgdl=list(LEVEL_EDGES), level_names=list(LEVEL_NAMES),
                     test_coverage={f"{r.coverage_target:g}": round(float(r.coverage), 4)
                                    for r in res[res.group == "all"].itertuples()})
    models = resolve_path(cfg["paths"]["models_dir"])
    (models / FILE).write_text(json.dumps(intervals, indent=1), encoding="utf-8")
    docs = resolve_path(cfg["paths"]["docs_dir"])
    res.round(4).to_csv(docs / "twin_intervals.csv", index=False)
    write_report(res, intervals, cal.patient_id.nunique(), test.patient_id.nunique(), docs / "twin_intervals.md")
    print(res.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s -> {models / FILE}, {docs / 'twin_intervals.md'}")


if __name__ == "__main__":
    main()
