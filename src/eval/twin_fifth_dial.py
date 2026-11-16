"""Experiment: does a fifth twin dial (counter-regulation strength) earn its place?

    python -m src.eval.twin_fifth_dial [--limit N]

The jury asked whether four physiological dials are enough. The rule we committed to (finals feedback doc):
add a dial only if (1) the data can identify it and (2) it lowers error on days the twin has not seen.
Counter-regulation (glucagon / adrenaline defence against lows, `cr_strength` in the simulator) is the
candidate with the clearest link to lows. It acts only when glucose falls below ~75 mg/dL, so it is fitted only
for patients on insulin or a sulfonylurea (as drug sensitivity is).

VALIDATION patients only (the test patients stay untouched for any later decision). Each patient is fitted twice
on days 1-7, with the deployed four dials and with four + counter-regulation, then both are scored on days 8-14:
open-loop CGM error, time-below-70 error, agreement on 2 h low windows, and 2 h projections from the observer.
Recovery compares the fitted `cr_strength` with the hidden simulator truth. Paired 95% CIs resample patients.
Writes docs/twin_fifth_dial.md and docs/twin_fifth_dial.csv. SYNTHETIC DATA, NOT CLINICALLY VALIDATED.
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import load_config, resolve_path
from src.eval import twin_validation as TV
from src.twin import personalise as P

FIVE = (*P.FIT_PARAMS, "cr_strength")
SETS = ("four", "five")


def evaluate_patient(cfg: dict[str, Any], pid: str, hidden: pd.DataFrame) -> dict[str, Any]:
    dt = cfg["sampling"]["interval_minutes"]
    cal = cfg["simulation"]["calibration"]
    warm = cfg["simulation"]["warmup_days"] * 24 * 60 // dt
    per_day = 24 * 60 // dt
    horizon = cfg["windows"]["horizon_hours"] * 60 // dt
    stride = cfg["windows"]["stride_minutes"] // dt
    inp = P.patient_inputs(cfg, pid)
    n_keep = len(inp.series)
    st = P.settings(cfg, n_keep // per_day)
    default = P.ehr_default_params(pd.DataFrame([{**inp.ehr_row.to_dict(), "patient_id": pid}]), cal, dt)
    s = P.schedule_from_log(inp, dt, warm)
    cgm = inp.series[:, 0]
    on_drug = P.on_glucose_lowering_drug(inp.ehr_row)
    fits = {name: P.fit_parameters(s, cgm, default, cal, dt, warm, st["fit_days"] * per_day, on_drug,
                                   st["prior_weight"], seed=cfg["seed"], params=prm)
            for name, prm in (("four", P.FIT_PARAMS), ("five", FIVE))}
    lo = st["fit_days"] * per_day
    m = np.zeros(n_keep, bool)
    m[lo:] = True
    m &= ~np.isnan(cgm)
    sim = P.sensor_view(P.simulate_schedule(pd.concat([fits[n].params for n in SETS], ignore_index=True), s, dt, cal), dt)[:, warm:]
    o = cgm[m]
    out: dict[str, Any] = {"patient_id": pid, "regimen": inp.ehr_row["regimen"], "on_drug": on_drug,
                           "obs_tbr": float(np.mean(o < 70))}
    anchors = np.arange(lo + per_day // 24, n_keep - horizon, TV.FORECAST_STRIDE_H * 60 // dt)
    anchors = anchors[[np.isfinite(cgm[max(0, a - 5): a + 1]).any() for a in anchors]]
    truth = np.stack([cgm[a + 1: a + 1 + horizon] for a in anchors])
    for j, name in enumerate(SETS):
        x = sim[j, m]
        out[f"{name}_mae"] = float(np.mean(np.abs(x - o)))
        out[f"{name}_tbr_err"] = abs(float(np.mean(x < 70)) - out["obs_tbr"])
        ev = TV._window_events(cgm, sim[j], lo, n_keep, stride, horizon, cfg["thresholds"], cfg["windows"]["min_horizon_coverage"])
        pos = ev.obs_hypo.to_numpy(bool)
        out[f"{name}_hypo_windows"] = int(pos.sum())
        out[f"{name}_hypo_hits"] = int((pos & ev.sim_hypo.to_numpy(bool)).sum())
        out[f"{name}_hypo_false"] = int((~pos & ev.sim_hypo.to_numpy(bool)).sum())
        fc = TV._forecasts(fits[name].params, s, cgm, anchors, warm, horizon, dt, cal, st["observer_gain"])
        out[f"{name}_fc_mae"] = float(np.nanmean(np.abs(fc - truth)))
        out[f"{name}_seconds"] = float(fits[name].info["seconds"])
    true = hidden.loc[pid]
    for n in FIVE:
        out[f"true_{n}"] = float(true[n])
        out[f"ehr_{n}"] = float(default.iloc[0][n])
        for name in SETS:
            out[f"{name}_{n}"] = float(fits[name].params.iloc[0][n])
    return out


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    d = a - b
    rng = np.random.default_rng(seed)
    boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    return float(np.mean(d)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rel_err(est: pd.Series, true: pd.Series) -> float:
    return float(np.median(np.abs(est / true - 1)))


def write_report(per: pd.DataFrame, cfg: dict[str, Any], path) -> dict[str, Any]:
    n_boot, seed = cfg["eval"]["n_bootstrap"], cfg["seed"]
    drug = per[per.on_drug]
    rows, verdict = [], {}
    for label, col, sub in (("Held-out CGM error, all patients (mg/dL)", "mae", per),
                            ("Held-out CGM error, patients on insulin or a sulfonylurea (mg/dL)", "mae", drug),
                            ("Time-below-70 error, on-drug patients (percentage points)", "tbr_err", drug),
                            ("2 h projection error, on-drug patients (mg/dL)", "fc_mae", drug)):
        scale = 100 if col == "tbr_err" else 1
        d, lo_, hi_ = paired_ci(sub[f"five_{col}"].to_numpy() * scale, sub[f"four_{col}"].to_numpy() * scale, n_boot, seed)
        rows.append(f"| {label} | {sub[f'four_{col}'].mean() * scale:.2f} | {sub[f'five_{col}'].mean() * scale:.2f} | "
                    f"{d:+.2f} ({lo_:+.2f} to {hi_:+.2f}) |")
        verdict[f"{col}_{len(sub)}"] = (d, lo_, hi_)
    hits = {n: drug[f"{n}_hypo_hits"].sum() / max(drug[f"{n}_hypo_windows"].sum(), 1) for n in SETS}
    false = {n: int(drug[f"{n}_hypo_false"].sum()) for n in SETS}
    cr = drug
    rec_default, rec_fit = rel_err(cr.ehr_cr_strength, cr.true_cr_strength), rel_err(cr.five_cr_strength, cr.true_cr_strength)
    rho_default = float(spearmanr(cr.ehr_cr_strength, cr.true_cr_strength).statistic)
    rho_fit = float(spearmanr(cr.five_cr_strength, cr.true_cr_strength).statistic)
    shift = {n: (rel_err(drug[f"four_{n}"], drug[f"true_{n}"]), rel_err(drug[f"five_{n}"], drug[f"true_{n}"])) for n in P.FIT_PARAMS}
    mae_d, mae_lo, mae_hi = verdict[f"mae_{len(drug)}"]
    lowers_error = mae_hi < 0
    identifiable = rec_fit < rec_default and rho_fit > rho_default
    adopt = lowers_error and identifiable
    lines = [
        "# Experiment: a fifth twin dial (counter-regulation)", "",
        "**SYNTHETIC DATA. Not clinically validated.** Generated by `python -m src.eval.twin_fifth_dial`. "
        f"{len(per)} validation patients ({len(drug)} on insulin or a sulfonylurea); test patients untouched.", "",
        "**Question:** are the four deployed dials (basal glucose, carb sensitivity, clearance time, drug sensitivity) "
        "enough, or does fitting counter-regulation strength, the body's glucagon and adrenaline defence against lows, "
        "improve the twin? Rule fixed in advance: adopt it only if it **lowers error on held-out days** and the data "
        "**can identify it** (the fitted value is closer to the hidden truth than the health-record default).", "",
        f"## Verdict: **{'adopt the fifth dial' if adopt else 'keep four dials'}**", "",
        f"* Lowers held-out error for on-drug patients: **{'yes' if lowers_error else 'no'}** "
        f"({mae_d:+.2f} mg/dL, 95% CI {mae_lo:+.2f} to {mae_hi:+.2f}).",
        f"* Identifiable from one week of CGM: **{'yes' if identifiable else 'no'}** (median relative error "
        f"{rec_default:.1%} for the record default vs {rec_fit:.1%} fitted; Spearman with truth {rho_default:.2f} vs {rho_fit:.2f}).", "",
        "## Held-out days 8-14 (five dials minus four dials; negative = five is better)", "",
        "| Measure | Four dials | Five dials | Difference (95% CI) |", "|---|---|---|---|", *rows, "",
        f"Low (< 70 mg/dL) 2 h windows reproduced by the twin, on-drug patients: four dials {hits['four']:.0%}, "
        f"five dials {hits['five']:.0%} (false low windows: {false['four']} vs {false['five']}).", "",
        "## Does adding the dial disturb the other four? (median relative error vs hidden truth, on-drug patients)", "",
        "| Dial | Four-dial fit | Five-dial fit |", "|---|---|---|",
        *[f"| `{n}` | {a:.1%} | {b:.1%} |" for n, (a, b) in shift.items()], "",
        "## Reading this", "",
        "* Counter-regulation only acts when glucose falls below about 75 mg/dL. With few lows in a week of data, there "
        "is little signal to pin it down, which is the identifiability problem in practice.",
        "* A dial that does not lower held-out error adds a number a clinician might over-read. The rule keeps the twin "
        "at four dials unless the evidence says otherwise.",
        "* On real patients, lows are faster and more frequent (docs/simulator_realism.md), which may give this dial more "
        "signal; the experiment should be repeated on real linked data with meal and dose logs.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"adopt": adopt, "lowers_error": lowers_error, "identifiable": identifiable}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N validation patients (quick check)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw, proc, docs = (resolve_path(cfg["paths"][k]) for k in ("raw_dir", "processed_dir", "docs_dir"))
    hidden = pd.read_csv(raw / "patient_params_HIDDEN.csv").set_index("patient_id")
    pids = sorted(pd.read_csv(proc / "splits.csv").query("split == 'val'").patient_id)[:args.limit]
    rows = []
    for i, pid in enumerate(pids):
        r = evaluate_patient(cfg, pid, hidden)
        rows.append(r)
        print(f"[{i + 1}/{len(pids)}] {pid} on_drug={r['on_drug']}  held-out MAE four {r['four_mae']:.2f}  five {r['five_mae']:.2f}",
              flush=True)
    per = pd.DataFrame(rows)
    per.round(5).to_csv(docs / "twin_fifth_dial.csv", index=False)
    print(write_report(per, cfg, docs / "twin_fifth_dial.md"))


if __name__ == "__main__":
    main()
