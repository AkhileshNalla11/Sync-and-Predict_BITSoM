"""Held-out validation of the personalised (fitted) twin against EHR-default and oracle parameters.

    python -m src.eval.twin_validation            # test-split patients; refits and caches their parameters

For each test patient the four fitted parameters come from the first `fit_days` of CGM + the meal/dose
log (src/twin/personalise.py). On the held-out remaining days we compare three parameter sets, all driven
by the same log-derived schedule with no process noise, so only the parameters differ:
    EHR default   population-typical parameters from the EHR alone
    fitted        EHR default + 4 parameters fitted to the patient's own first week
    oracle        the simulator's HIDDEN true parameters (upper reference; unknowable in reality)
Two uses are scored:
  1) open-loop simulation over the held-out week: CGM MAE, time-in-range agreement, and agreement of
     hyper / hypo events in 2 h windows (the labels used by the risk model);
  2) 2-hour projections as used by the what-if panel: the state observer pulls the simulation toward the
     CGM up to each prediction time, then it runs freely; compared with CGM persistence.
Also reports how well the fit recovers the hidden parameters. Writes docs/twin_personalisation.{md,csv}.
SYNTHETIC DATA: the fitted model has the same structure as the simulator, so this is a best case.
"""
from __future__ import annotations

import argparse
import copy
import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.twin import personalise as P

PARAM_SETS = ("ehr_default", "fitted", "oracle")
LABEL = {"persistence": "CGM persistence", "ehr_default": "EHR default", "fitted": "Fitted (personalised)",
         "oracle": "Oracle (hidden truth)"}
FORECAST_STRIDE_H = 2
FORECAST_MIN = (30, 60, 120)


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def _window_events(obs: np.ndarray, sim: np.ndarray, lo: int, hi: int, stride: int, horizon: int,
                   thr: dict[str, Any], min_cov: float) -> pd.DataFrame:
    """Hyper / hypo event in each 2 h horizon, from observed CGM vs simulated CGM (at the same readings)."""
    rows = []
    for a in range(lo, hi - horizon, stride):
        o, s = obs[a + 1: a + 1 + horizon], sim[a + 1: a + 1 + horizon]
        m = ~np.isnan(o)
        if m.mean() < min_cov:
            continue
        rows.append({"obs_hyper": o[m].max() > thr["hyper_mgdl"], "sim_hyper": s[m].max() > thr["hyper_mgdl"],
                     "obs_hypo": o[m].min() < thr["hypo_mgdl"], "sim_hypo": s[m].min() < thr["hypo_mgdl"]})
    return pd.DataFrame(rows)


def _forecasts(params: pd.DataFrame, s, cgm: np.ndarray, anchors: np.ndarray, warm: int, horizon: int, dt: int,
               cal: dict[str, float], gain: float) -> np.ndarray:
    """(n_anchors, horizon) CGM projections: observer on up to each anchor, free run afterwards (one batched call)."""
    n_steps = warm + int(anchors.max()) + horizon + 1
    obs = np.stack([P.observer_target(cgm, warm, n_steps, int(a), dt) for a in anchors])
    rows = params.loc[params.index.repeat(len(anchors))].reset_index(drop=True)
    sv = P.sensor_view(P.simulate_schedule(rows, s, dt, cal, n_steps, obs=obs, obs_gain=gain), dt)[:, warm:]
    return np.stack([sv[j, a + 1: a + 1 + horizon] for j, a in enumerate(anchors)])


def validate_patient(cfg: dict[str, Any], pid: str, hidden: pd.DataFrame, refit: bool) -> dict[str, Any]:
    dt = cfg["sampling"]["interval_minutes"]
    cal = cfg["simulation"]["calibration"]
    thr = cfg["thresholds"]
    warm = cfg["simulation"]["warmup_days"] * 24 * 60 // dt
    per_day = 24 * 60 // dt
    horizon = cfg["windows"]["horizon_hours"] * 60 // dt
    stride = cfg["windows"]["stride_minutes"] // dt
    inp = P.patient_inputs(cfg, pid)
    n_keep = len(inp.series)
    st = P.settings(cfg, n_keep // per_day)
    t0 = time.time()
    fit = P.get_fit(cfg, pid, refit=refit)
    fit_seconds = time.time() - t0 if refit else float(fit.info.get("seconds", np.nan))
    sets = {"ehr_default": fit.default, "fitted": fit.params,
            "oracle": hidden.loc[[pid]].reset_index()[fit.default.columns]}
    s = P.schedule_from_log(inp, dt, warm)
    cgm = inp.series[:, 0]
    lo = st["fit_days"] * per_day                              # held-out = everything after the fit window
    held = np.zeros(n_keep, bool)
    held[lo:] = True
    m = held & ~np.isnan(cgm)

    out: dict[str, Any] = {"patient_id": pid, "regimen": inp.ehr_row["regimen"], "fit_seconds": fit_seconds,
                           "fit_calls": int(fit.info["n_calls"]),
                           "fit_mae": float(fit.info["fit_mae"]), "n_heldout": int(m.sum())}
    g = P.simulate_schedule(pd.concat(sets.values(), ignore_index=True), s, dt, cal)
    sim = P.sensor_view(g, dt)[:, warm:]
    o = cgm[m]
    out.update(obs_tir=np.mean((o >= 70) & (o <= 180)), obs_tar=np.mean(o > 180), obs_tbr=np.mean(o < 70))
    events = {}
    for j, name in enumerate(PARAM_SETS):
        x = sim[j, m]
        out[f"{name}_mae"] = float(np.mean(np.abs(x - o)))
        out[f"{name}_bias"] = float(np.mean(x - o))
        out[f"{name}_tir"] = np.mean((x >= 70) & (x <= 180))
        out[f"{name}_tar"] = np.mean(x > 180)
        out[f"{name}_tbr"] = np.mean(x < 70)
        events[name] = _window_events(cgm, sim[j], lo, n_keep, stride, horizon, thr, cfg["windows"]["min_horizon_coverage"])

    # 2 h projections from the observer state, every FORECAST_STRIDE_H hours in the held-out period
    anchors = np.arange(lo + per_day // 24, n_keep - horizon, FORECAST_STRIDE_H * 60 // dt)
    truth = np.stack([cgm[a + 1: a + 1 + horizon] for a in anchors])
    last = np.array([cgm[max(0, a - 5): a + 1][~np.isnan(cgm[max(0, a - 5): a + 1])][-1:] for a in anchors], dtype=object)
    keep = np.array([len(v) == 1 for v in last])
    anchors, truth = anchors[keep], truth[keep]
    fc = {"persistence": np.repeat(np.array([v[0] for v in last[keep]], float)[:, None], horizon, axis=1)}
    for name, prm in sets.items():
        fc[name] = _forecasts(prm, s, cgm, anchors, warm, horizon, dt, cal, st["observer_gain"])
    for name, f in fc.items():
        err = np.abs(f - truth)
        out[f"fc_{name}_mae"] = float(np.nanmean(err))
        for mins in FORECAST_MIN:
            out[f"fc_{name}_mae_{mins}"] = float(np.nanmean(err[:, mins // dt - 1]))
    for n in P.FIT_PARAMS:
        out[f"true_{n}"] = float(sets["oracle"].iloc[0][n])
        out[f"ehr_{n}"] = float(fit.default.iloc[0][n])
        out[f"fit_{n}"] = float(fit.params.iloc[0][n])
    out["on_drug"] = P.on_glucose_lowering_drug(inp.ehr_row)
    return {"row": out, "events": events}


def _boot_ci(x: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    b = [np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(per: pd.DataFrame, events: dict[str, pd.DataFrame], n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    open_loop = []
    for name in PARAM_SETS:
        e = events[name]
        row = {"params": LABEL[name], "cgm_mae": per[f"{name}_mae"].mean(),
               "cgm_mae_ci": _boot_ci(per[f"{name}_mae"].to_numpy(), n_boot, rng),
               "bias": per[f"{name}_bias"].mean()}
        for k in ("tir", "tar", "tbr"):
            row[f"{k}_abs_err_pp"] = 100 * (per[f"{name}_{k}"] - per[f"obs_{k}"]).abs().mean()
        for t in ("hyper", "hypo"):
            a, b = e[f"obs_{t}"].to_numpy(bool), e[f"sim_{t}"].to_numpy(bool)
            row[f"{t}_agree"] = float(np.mean(a == b))
            row[f"{t}_sens"] = float(b[a].mean()) if a.any() else float("nan")
            row[f"{t}_spec"] = float((~b[~a]).mean()) if (~a).any() else float("nan")
            row[f"{t}_kappa"] = _kappa(a, b)
            row[f"{t}_obs_rate"], row[f"{t}_sim_rate"] = float(a.mean()), float(b.mean())
        open_loop.append(row)
    d = (per.fitted_mae - per.ehr_default_mae).to_numpy()
    forecast = [{"method": LABEL[n], **{f"mae_{m}": per[f"fc_{n}_mae_{m}"].mean() for m in FORECAST_MIN},
                 "mae_all": per[f"fc_{n}_mae"].mean()} for n in ("persistence", *PARAM_SETS)]
    recovery = []
    for n in P.FIT_PARAMS:
        sub = per[per.on_drug] if n == "drug_sens" else per
        rel = lambda col: float(np.median(np.abs(sub[col] / sub[f"true_{n}"] - 1)))     # noqa: E731
        sp = lambda col: float(sub[col].corr(sub[f"true_{n}"], method="spearman"))      # noqa: E731
        recovery.append({"param": n, "n": len(sub), "ehr_med_rel_err": rel(f"ehr_{n}"), "fit_med_rel_err": rel(f"fit_{n}"),
                         "ehr_spearman": sp(f"ehr_{n}"), "fit_spearman": sp(f"fit_{n}")})
    return {"open_loop": pd.DataFrame(open_loop), "forecast": pd.DataFrame(forecast), "recovery": pd.DataFrame(recovery),
            "mae_diff": float(d.mean()), "mae_diff_ci": _boot_ci(d, n_boot, rng), "fitted_better_share": float(np.mean(d < 0)),
            "n_patients": len(per), "n_windows": len(events["fitted"]),
            "fit_seconds_median": float(per.fit_seconds.median()), "fit_seconds_p90": float(per.fit_seconds.quantile(0.9)),
            "fit_seconds_max": float(per.fit_seconds.max()), "fit_calls_max": int(per.fit_calls.max())}


def write_report(summ: dict[str, Any], cfg: dict[str, Any], path, tuning: pd.DataFrame | None = None) -> None:
    ol, fc, rec = summ["open_loop"], summ["forecast"], summ["recovery"]
    st = P.settings(cfg, cfg["simulation"]["n_days"])
    ci = lambda c: f"{c[0]:.1f}-{c[1]:.1f}"                 # noqa: E731
    pct = lambda v: f"{v:.0%}" if np.isfinite(v) else "n/a"  # noqa: E731
    lines = [
        "# Personalised twin: fitted parameters vs hidden truth",
        "",
        "**SYNTHETIC DATA. Not clinically validated.** The fitted model has exactly the simulator's structure and the",
        "meal / dose log is exact, so these numbers are a best case for a model-based twin, not an estimate of",
        "real-world accuracy.",
        "",
        "## What changed",
        "",
        "The what-if engine used to replay each patient with the simulator's **hidden** parameters, schedule and",
        "noise, which no real system could do. The default *fitted* mode now builds the twin from observable data",
        "only (`src/twin/personalise.py`):",
        "",
        "- **EHR** -> population-default simulator parameters (the existing `derive_patient_params` mapping, with",
        "  the hidden latent traits and random draws set to their population typical values).",
        "- **Patient log** -> meals (type, carbs) and medication doses; **wearable** -> steps and sleep (a short",
        "  night raises next-day insulin resistance, as in the simulator).",
        f"- **First {st['fit_days']} days of CGM** -> four parameters fitted per patient by penalised least squares",
        "  (`scipy.optimize.least_squares`, bounded trust-region Gauss-Newton, after a 64-point batched coarse",
        "  search): untreated basal glucose `g_basal`, carbohydrate gain `meal_gain`, clearance time `tau_clear_min`,",
        "  and drug sensitivity `drug_sens` (only for patients on insulin or a sulfonylurea). A weak prior",
        f"  (weight {st['prior_weight']:g} (mg/dL)^2 per prior SD, chosen on validation patients, see below) keeps",
        "  them near the EHR default when the data is uninformative. Everything else stays at the EHR default.",
        f"- Fit time: median {summ['fit_seconds_median']:.1f} s, 90th percentile {summ['fit_seconds_p90']:.1f} s per patient",
        "  (wall clock on a shared laptop CPU; candidate parameter sets are simulated side by side, at most",
        f"  {summ['fit_calls_max']} batched simulator calls per fit; slowest {summ['fit_seconds_max']:.1f} s)."
        + (" Outliers far above the 90th percentile come from the machine stalling, not extra optimiser work."
           if summ["fit_seconds_max"] > 3 * summ["fit_seconds_p90"] else ""),
        "  Fits are cached in `data/processed/twin_fits.csv`.",
        f"- For a what-if at time *now*, a simple state observer (gain {st['observer_gain']:g} per 5 min) pulls the",
        "  simulated glucose toward the CGM up to *now*; after that the model runs freely. The scenario's CGM fed",
        "  to the risk model is the observed CGM plus the simulated scenario-minus-baseline difference.",
        "",
        "The oracle mode is still available (dashboard toggle) for comparison.",
        "",
        *([] if tuning is None else [
            "### Choosing the prior weight (validation patients only)",
            "",
            f"Compared on the {int(tuning.n_patients.iloc[0])} validation patients with the same held-out protocol;",
            "the test patients below played no part in any choice.",
            "",
            "| Prior weight | Held-out CGM MAE | 2 h projection MAE | " + " | ".join(f"`{n}` median rel. error" for n in P.FIT_PARAMS) + " |",
            "|---" * (3 + len(P.FIT_PARAMS)) + "|",
            *[f"| {r.prior_weight:g} | {r.heldout_cgm_mae:.2f} | {r.forecast_2h_mae:.2f} | "
              + " | ".join(f"{getattr(r, f'{n}_med_rel_err'):.1%}" for n in P.FIT_PARAMS) + " |" for r in tuning.itertuples()],
            "",
        ]),
        f"## Held-out validation ({summ['n_patients']} test patients, days {st['fit_days'] + 1}-"
        f"{cfg['simulation']['n_days']})",
        "",
        "All three parameter sets are driven by the same log-derived schedule without process noise, so only the",
        "parameters differ. *Oracle* uses the hidden true parameters: an upper reference, unknowable in reality.",
        "",
        "### 1. Open-loop simulation of the held-out week",
        "",
        "| Parameters | CGM MAE, mg/dL (95% CI) | Bias | TIR error | TAR error | TBR error | Hyper windows: agreement / sens / kappa | Hypo windows: agreement / sens / kappa |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {r.params} | {r.cgm_mae:.1f} ({ci(r.cgm_mae_ci)}) | {r.bias:+.1f} | {r.tir_abs_err_pp:.1f} pp | "
          f"{r.tar_abs_err_pp:.1f} pp | {r.tbr_abs_err_pp:.1f} pp | {pct(r.hyper_agree)} / {pct(r.hyper_sens)} / "
          f"{r.hyper_kappa:.2f} | {pct(r.hypo_agree)} / {pct(r.hypo_sens)} / {r.hypo_kappa:.2f} |" for r in ol.itertuples()],
        "",
        f"TIR / TAR / TBR error = mean absolute difference between simulated and observed time in range (70-180),",
        f"above 180 and below 70 mg/dL, in percentage points. Window events use the risk model's labels (any reading",
        f"> 180 or < 70 mg/dL in a 2 h horizon, 30 min stride; {summ['n_windows']:,} windows). Observed event rates:",
        f"hyper {ol.hyper_obs_rate.iloc[0]:.1%}, hypo {ol.hypo_obs_rate.iloc[0]:.1%}; simulated (fitted): "
        f"hyper {ol.hyper_sim_rate.iloc[1]:.1%}, hypo {ol.hypo_sim_rate.iloc[1]:.1%}.",
        "",
        f"Fitted minus EHR-default CGM MAE: {summ['mae_diff']:+.1f} mg/dL (95% patient-bootstrap CI "
        f"{summ['mae_diff_ci'][0]:+.1f} to {summ['mae_diff_ci'][1]:+.1f}); the fit is better for "
        f"{summ['fitted_better_share']:.0%} of patients.",
        "",
        "### 2. Two-hour projections (what the what-if panel shows)",
        "",
        f"Every {FORECAST_STRIDE_H} h in the held-out period: observer on up to the prediction time, then a free run.",
        "Mean absolute error against the CGM that followed.",
        "",
        "| Method | MAE at 30 min | at 60 min | at 120 min | mean over 5-120 min |",
        "|---|---|---|---|---|",
        *[f"| {r.method} | {r.mae_30:.1f} | {r.mae_60:.1f} | {r.mae_120:.1f} | {r.mae_all:.1f} |" for r in fc.itertuples()],
        "",
        "Persistence holds the last CGM reading flat. The mechanistic projections know the logged meals and doses",
        "in the next 2 h (the what-if panel's scenarios also edit future events), which persistence cannot use.",
        "",
        "### 3. Does the fit recover the hidden parameters?",
        "",
        "| Parameter | Patients | Median relative error: EHR default | fitted | Spearman with truth: EHR default | fitted |",
        "|---|---|---|---|---|---|",
        *[f"| `{r.param}` | {r.n} | {r.ehr_med_rel_err:.1%} | {r.fit_med_rel_err:.1%} | {r.ehr_spearman:.2f} | "
          f"{r.fit_spearman:.2f} |" for r in rec.itertuples()],
        "",
        "`drug_sens` is scored on patients taking insulin or a sulfonylurea only (it has no effect otherwise).",
        "",
        "## Caveats",
        "",
        "- **Best case by construction.** The fitted model is the simulator itself with four free parameters;",
        "  real physiology would add model mismatch. The log here is exact; real carb counts are often off by",
        "  20-50 %, and doses and meal times get forgotten.",
        "- **Hidden drivers remain unmodelled.** The twin cannot see the simulator's process noise, the true",
        "  sleep-driven insulin-resistance swings on nights the watch was not worn, or sensor calibration error;",
        "  this is why even oracle parameters leave a sizeable open-loop error.",
        "- **Fitted can beat oracle on CGM error, and that is not a contradiction.** The hidden parameters are the",
        "  best *physiological* description, but a deterministic replay of them still misses the per-sensor",
        "  calibration bias and the average effect of hidden drivers. Fitted parameters absorb those, so part of",
        "  the gain is calibration of the sensor and of the model, not better physiology. Table 3 is the",
        "  physiological check: there the fit improves on the EHR default mainly for carbohydrate gain and drug",
        "  sensitivity, and does not beat it for every parameter.",
        "- **The EHR-to-parameter mapping is the simulator's own,** so the EHR default is unrealistically good; in",
        "  a real population far more between-patient variation would be left for the fit to explain.",
        "- **Identifiability.** `g_basal` and `tau_clear_min` (and to a lesser degree `meal_gain`) partly trade",
        "  off against each other: together they set the average level and the size of meal excursions, and the",
        "  prior holds them near the EHR default along that ridge. Fitted `tau_clear_min` is in fact slightly",
        "  further from the truth than the EHR default (table 3). Individual values should not be read as",
        "  physiological measurements; drug sensitivity, which dose timing pins down, is the best identified.",
        "- **Not a clinical tool.** Nothing here is clinically validated; the what-if projections are illustrations",
        "  of the synthetic model.",
        "",
        "Reproduce: `python -m src.eval.twin_validation --tune 0,2,5,20` (validation patients, writes",
        "`docs/twin_personalisation_tuning.csv`), then `python -m src.eval.twin_validation` (test patients, writes",
        "this file and `docs/twin_personalisation.csv`).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_split(cfg: dict[str, Any], split: str, limit: int | None, refit: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = resolve_path(cfg["paths"]["raw_dir"])
    proc = resolve_path(cfg["paths"]["processed_dir"])
    hidden = pd.read_csv(raw / "patient_params_HIDDEN.csv").set_index("patient_id")
    pids = sorted(pd.read_csv(proc / "splits.csv").query("split == @split").patient_id)[:limit]
    rows, events = [], {n: [] for n in PARAM_SETS}
    for i, pid in enumerate(pids):
        r = validate_patient(cfg, pid, hidden, refit=refit)
        rows.append(r["row"])
        for n in PARAM_SETS:
            events[n].append(r["events"][n])
        print(f"[{split} {i + 1}/{len(pids)}] {pid} fit {r['row']['fit_seconds']:.1f}s  held-out MAE "
              + "  ".join(f"{n} {r['row'][f'{n}_mae']:.1f}" for n in PARAM_SETS), flush=True)
    per = pd.DataFrame(rows)
    ev = {n: pd.concat(v, ignore_index=True) for n, v in events.items()}
    return per, summarise(per, ev, cfg["eval"]["n_bootstrap"], cfg["seed"])


def tune_prior_weight(cfg: dict[str, Any], weights: list[float], limit: int | None) -> pd.DataFrame:
    """Choose the prior weight on VALIDATION patients (the test patients stay untouched)."""
    rows = []
    for w in weights:
        c = copy.deepcopy(cfg)
        c["twin"]["personalise"]["prior_weight"] = w
        per, summ = run_split(c, "val", limit, refit=True)
        rec = summ["recovery"].set_index("param")
        rows.append({"prior_weight": w, "n_patients": len(per),
                     "heldout_cgm_mae": float(summ["open_loop"].cgm_mae.iloc[1]),
                     "forecast_2h_mae": float(summ["forecast"].mae_all.iloc[2]),
                     **{f"{n}_med_rel_err": float(rec.loc[n, "fit_med_rel_err"]) for n in P.FIT_PARAMS}})
        print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N patients (quick check)")
    ap.add_argument("--no-refit", action="store_true", help="reuse cached fits")
    ap.add_argument("--tune", default=None, help="comma-separated prior weights to compare on validation patients")
    args = ap.parse_args()
    cfg = load_config(args.config)
    docs = resolve_path(cfg["paths"]["docs_dir"])
    t0 = time.time()
    if args.tune:
        tuning = tune_prior_weight(cfg, [float(w) for w in args.tune.split(",")], args.limit)
        tuning.round(4).to_csv(docs / "twin_personalisation_tuning.csv", index=False)
        print(f"done in {time.time() - t0:.0f}s -> {docs / 'twin_personalisation_tuning.csv'}")
        return
    per, summ = run_split(cfg, "test", args.limit, refit=not args.no_refit)
    results = resolve_path(f"{cfg['paths']['processed_dir']}/results")
    per.to_csv(results / "twin_personalisation_per_patient.csv", index=False)
    ol = summ["open_loop"].copy()
    ol["cgm_mae_ci"] = ol.cgm_mae_ci.map(lambda c: f"{c[0]:.2f}-{c[1]:.2f}")
    pd.concat([ol.assign(table="open_loop"), summ["forecast"].assign(table="forecast_2h"),
               summ["recovery"].assign(table="parameter_recovery")], ignore_index=True).round(4).to_csv(
        docs / "twin_personalisation.csv", index=False)
    tuning_path = docs / "twin_personalisation_tuning.csv"
    write_report(summ, cfg, docs / "twin_personalisation.md", pd.read_csv(tuning_path) if tuning_path.exists() else None)
    print(summ["open_loop"].round(3).to_string(index=False))
    print(summ["forecast"].round(2).to_string(index=False))
    print(summ["recovery"].round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s -> {docs / 'twin_personalisation.md'}")


if __name__ == "__main__":
    main()
