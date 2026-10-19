"""How realistic is the simulator? Its CGM against real CGM from two public cohorts, metric by metric.

    python -m src.eval.simulator_realism [--shanghai-dir data/external/shanghai] [--cgmacros-dir data/external/cgmacros]

"Simulated glucose is smoother than real glucose" was a caveat; this script measures it. For every patient
(ShanghaiT2DM: every recording period) the same statistics are computed on a common 15-minute grid, because
ShanghaiT2DM samples every 15 min (synthetic and CGMacros 5-min series are subsampled; ShanghaiT2DM's grid
points at its native readings are kept, so no interpolated value enters):
  * level and spread: mean, CV, time in / above / below range;
  * events: hypo (< 70) and hyper (> 180) episodes per day (>= 2 consecutive readings past the threshold),
    and the share of hypo episodes starting at night (22:00-06:00);
  * dynamics: SD and 95th percentile of the 15-min change, and how often the direction of change reverses
    (a roughness measure: a smooth curve rarely reverses);
  * how lows arrive: glucose 60 min before a hypo onset and the mean fall rate over that hour;
  * daily rhythm: correlation of the cohort's hourly deviation-from-mean profile with the real cohorts';
  * lows by drug class (insulin / sulfonylurea without insulin / neither), where the EHR allows.
Verdict per metric: the synthetic cohort median against the interquartile range of the pooled real cohorts.
Rare-event metrics (time below 70, hypo episodes per day, share of patients with any low) are mostly zero per
patient, so they use cohort means instead, judged against the span of the two real cohorts' means.
Writes docs/simulator_realism.{md,csv} and docs/figures/fig_simulator_realism.png. Aggregate numbers only.
Real cohorts are not the target population (Chinese hospital cohort; small US cohort): they bound realism,
they do not define it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ROOT, load_config, resolve_path  # noqa: E402
from src.data.build_dataset import load_processed  # noqa: E402
from src.eval.metrics import event_episodes  # noqa: E402

STEP_MIN = 15
PRE_STEPS = 8                                   # 2 h of trajectory before a hypo onset (8 x 15 min)
SOURCES = ("Synthetic", "ShanghaiT2DM", "CGMacros T2D")
COLORS = {"Synthetic": "#0f766e", "ShanghaiT2DM": "#eb6834", "CGMacros T2D": "#2a78d6"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
METRICS = [  # key, label, format
    ("mean", "Mean glucose (mg/dL)", "{:.0f}"), ("cv", "Coefficient of variation (%)", "{:.1f}"),
    ("tir", "Time in range 70-180 (%)", "{:.1f}"), ("tar", "Time above 180 (%)", "{:.1f}"),
    ("tbr", "Time below 70 (%), cohort mean", "{:.2f}"), ("hypo_per_day", "Hypo episodes per day, cohort mean", "{:.2f}"),
    ("any_hypo", "Patients with any low episode (%)", "{:.0f}"),
    ("hyper_per_day", "Hyper episodes per day", "{:.2f}"),
    ("delta_sd", "SD of 15-min change (mg/dL)", "{:.1f}"), ("delta_p95", "Large swings: 95th percentile of the absolute 15-min change (mg/dL)", "{:.1f}"),
    ("reversal_rate", "Direction reversals (% of steps)", "{:.1f}"),
    ("pre_hypo_g60", "Glucose 60 min before a low (mg/dL)", "{:.0f}"),
    ("pre_hypo_slope", "Fall rate in the hour before a low (mg/dL/min)", "{:.2f}"),
]


def to_15min(s: pd.Series) -> pd.Series:
    """Every third point of a regular 5-min series (ShanghaiT2DM grids start at a native reading)."""
    return s.iloc[::STEP_MIN // 5]


def patient_metrics(s: pd.Series) -> tuple[dict[str, float], list[np.ndarray], pd.Series]:
    """Statistics for one 15-min series; also the pre-hypo trajectories and the hourly deviation profile."""
    v = s.to_numpy(float)
    g = v[~np.isnan(v)]
    days = len(g) * STEP_MIN / 1440
    out: dict[str, float] = {"days": days}
    if len(g) < 96:                                                   # under a day of readings
        return out, [], pd.Series(dtype=float)
    out.update(mean=g.mean(), cv=100 * g.std() / g.mean(), tir=100 * np.mean((g >= 70) & (g <= 180)),
               tar=100 * np.mean(g > 180), tbr=100 * np.mean(g < 70))
    d = np.diff(v)
    dv = d[~np.isnan(d)]
    out.update(delta_sd=dv.std(), delta_p95=np.quantile(np.abs(dv), 0.95))
    sign = np.sign(d)
    ok = ~np.isnan(d[:-1]) & ~np.isnan(d[1:]) & (sign[:-1] != 0) & (sign[1:] != 0)
    out["reversal_rate"] = 100 * np.mean(sign[:-1][ok] != sign[1:][ok]) if ok.any() else np.nan
    hypo = event_episodes(v[None], 70, below=True, min_samples=2, clear_samples=2)[0]
    hyper = event_episodes(v[None], 180, below=False, min_samples=2, clear_samples=2)[0]
    out.update(hypo_per_day=len(hypo) / days, hyper_per_day=len(hyper) / days, any_hypo=100.0 * (len(hypo) > 0))
    hours = s.index.hour.to_numpy()
    if len(hypo):
        out["hypo_night_share"] = float(np.mean((hours[hypo] >= 22) | (hours[hypo] < 6)))
    traj = [v[o - PRE_STEPS: o + 1] for o in hypo if o >= PRE_STEPS]
    traj = [t for t in traj if not np.isnan(t[[PRE_STEPS - 4, PRE_STEPS]]).any()]
    if traj:
        out["pre_hypo_g60"] = float(np.mean([t[PRE_STEPS - 4] for t in traj]))
        out["pre_hypo_slope"] = float(np.mean([(t[PRE_STEPS] - t[PRE_STEPS - 4]) / 60 for t in traj]))
    prof = pd.Series(v - np.nanmean(v), index=hours).groupby(level=0).mean()
    return out, traj, prof


def synthetic_series(cfg: dict[str, Any]) -> tuple[list[pd.Series], pd.DataFrame]:
    ds = load_processed(cfg)
    ts = ds["timestamps"]
    ehr = pd.read_csv(resolve_path(cfg["paths"]["raw_dir"]) / "ehr.csv").set_index("patient_id")
    series = [pd.Series(ds["series"][i, :, 0], index=ts) for i in range(len(ds["patient_ids"]))]
    drugs = ehr.loc[[str(p) for p in ds["patient_ids"]], ["on_insulin", "on_sulfonylurea"]].reset_index(drop=True)
    return series, drugs


def shanghai_series(data_dir: Path, dt: int) -> tuple[list[pd.Series], pd.DataFrame]:
    from src.data.shanghai import load_cohort
    periods, static, _ = load_cohort(data_dir, dt)
    return [p.cgm for p in periods], static.loc[[p.period_id for p in periods], ["on_insulin", "on_sulfonylurea"]].reset_index(drop=True)


def cgmacros_series(data_dir: Path, dt: int) -> tuple[list[pd.Series], pd.DataFrame]:
    from src.data.cgmacros import load_cohort
    ps = load_cohort(data_dir, statuses=("t2d",), dt_min=dt)
    return [p.cgm for p in ps], pd.DataFrame({"on_insulin": np.nan, "on_sulfonylurea": np.nan}, index=range(len(ps)))


def drug_group(row: pd.Series) -> str:
    if pd.isna(row.on_insulin):
        return "unknown"
    if row.on_insulin:
        return "insulin"
    return "sulfonylurea, no insulin" if row.on_sulfonylurea else "neither"


def collect(sources: dict[str, tuple[list[pd.Series], pd.DataFrame]]) -> tuple[pd.DataFrame, dict, dict]:
    rows, trajs, profs = [], {}, {}
    for name, (series, drugs) in sources.items():
        trajs[name], ps = [], []
        for i, s in enumerate(series):
            m, traj, prof = patient_metrics(to_15min(s))
            if "mean" not in m:
                continue
            rows.append({"source": name, "unit": i, "drug_group": drug_group(drugs.iloc[i]), **m})
            trajs[name] += traj
            ps.append(prof)
        profs[name] = pd.concat(ps, axis=1).mean(axis=1).reindex(range(24))
    return pd.DataFrame(rows), trajs, profs


MEAN_METRICS = ("tbr", "hypo_per_day", "any_hypo")


def summary_table(per: pd.DataFrame, profs: dict) -> pd.DataFrame:
    real = per[per.source != "Synthetic"]
    real_srcs = [s for s in SOURCES[1:] if s in set(per.source)]
    rows = []
    for key, label, fmt in METRICS:
        r = {"metric": label, "key": key}
        for src in SOURCES:
            x = per.loc[per.source == src, key].dropna()
            if not len(x):
                r[src] = (np.nan,) * 3 + (0,)
            elif key in MEAN_METRICS:
                r[src] = (x.mean(), np.nan, np.nan, len(x))
            else:
                r[src] = (x.median(), x.quantile(0.25), x.quantile(0.75), len(x))
        if key in MEAN_METRICS:
            means = [r[s][0] for s in real_srcs]
            lo, hi = min(means), max(means)
        else:
            lo, hi = real[key].dropna().quantile([0.25, 0.75])
        syn = r["Synthetic"][0]
        r["verdict"] = "within real range" if lo <= syn <= hi else ("too high" if syn > hi else "too low")
        if key == "pre_hypo_slope" and r["verdict"] == "too high":      # less negative = a gentler fall
            r["verdict"] = "too slow"
        r["fmt"] = fmt
        rows.append(r)
    real_prof = pd.concat([profs[s] for s in real_srcs], axis=1).mean(axis=1)
    corr = float(np.corrcoef(profs["Synthetic"].to_numpy(), real_prof.to_numpy())[0, 1])
    ref = (float(np.corrcoef(profs[real_srcs[0]].to_numpy(), profs[real_srcs[1]].to_numpy())[0, 1])
           if len(real_srcs) == 2 else np.nan)
    rows.append({"metric": "Daily rhythm: correlation of the hourly profile with the real cohorts'"
                           + (f" (the two real cohorts with each other: {ref:.2f})" if np.isfinite(ref) else ""),
                 "key": "rhythm_corr", "Synthetic": (corr, np.nan, np.nan, 24),
                 "verdict": "similar" if corr >= (ref if np.isfinite(ref) else 0.7) - 0.1 else "different", "fmt": "{:.2f}"})
    return pd.DataFrame(rows)


def drug_table(per: pd.DataFrame) -> pd.DataFrame:
    d = per[per.drug_group != "unknown"]
    return (d.groupby(["source", "drug_group"]).agg(units=("unit", "size"), hypo_per_day=("hypo_per_day", "mean"),
                                                    tbr=("tbr", "mean")).reset_index())


def plot(per: pd.DataFrame, trajs: dict, profs: dict, raw15: dict, path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    bins = np.linspace(-60, 60, 61)
    for src in SOURCES:
        if src not in raw15:
            continue
        ax[0].hist(raw15[src], bins=bins, density=True, histtype="step", lw=2, color=COLORS[src], label=src)
        p = profs[src]
        ax[1].plot(p.index, p.to_numpy(), lw=2, color=COLORS[src], label=src)
        if trajs[src]:
            t = np.nanmean(np.vstack(trajs[src]), axis=0)
            ax[2].plot(np.arange(-PRE_STEPS, 1) * STEP_MIN, t, lw=2, marker="o", ms=3, color=COLORS[src],
                       label=f"{src} ({len(trajs[src])} lows)")
    ax[0].set(yscale="log", xlabel="15-min change in glucose (mg/dL)", ylabel="density (log scale)",
              title="How fast glucose moves")
    ax[1].set(xlabel="hour of day", ylabel="mg/dL vs the patient's mean", title="Daily rhythm", xticks=range(0, 24, 3))
    ax[2].axhline(70, color=INK_2, lw=1, ls="--")
    ax[2].set(xlabel="minutes before the low begins", ylabel="mean CGM (mg/dL)", title="How lows arrive")
    for a in ax:
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
        a.title.set_color(INK)
        a.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def cell(v: tuple, fmt: str) -> str:
    med, lo, hi, n = v
    if not n or np.isnan(med):
        return "n/a"
    return fmt.format(med) + ("" if np.isnan(lo) else f" ({fmt.format(lo)} to {fmt.format(hi)})")


def write_report(tab: pd.DataFrame, drugs: pd.DataFrame, per: pd.DataFrame, path: Path) -> None:
    n = per.groupby("source").size()
    srcs = [s for s in SOURCES if s in n.index]
    lines = [
        "# How realistic is the simulator?", "",
        "**Synthetic vs real CGM, metric by metric.** Generated by `python -m src.eval.simulator_realism`. Real data:",
        "ShanghaiT2DM (CC BY 4.0) and CGMacros T2D (CC BY-NC-SA 4.0), never redistributed; this page holds aggregates only.", "",
        "All series are compared on a common 15-minute grid (ShanghaiT2DM's native interval). Values are the median",
        "and interquartile range across patients (ShanghaiT2DM: recording periods). The verdict compares the",
        "synthetic median with the interquartile range of the two real cohorts pooled. The real cohorts are not the",
        "target population (a partly hospitalised Chinese cohort and 14 US adults), so they bound realism rather",
        "than define it.", "",
        "| Metric | " + " | ".join(f"{s} (n={n[s]})" for s in srcs) + " | Synthetic vs real |",
        "|---|" + "---|" * len(srcs) + "---|",
    ]
    for r in tab.itertuples():
        vals = [cell(tab.loc[r.Index, s], r.fmt) if isinstance(tab.loc[r.Index, s], tuple) else "" for s in srcs]
        verdict = r.verdict if r.verdict in ("within real range", "similar") else f"**{r.verdict}**"
        lines.append(f"| {r.metric} | " + " | ".join(vals) + f" | {verdict} |")
    v = {r.key: r for r in tab.itertuples()}
    syn = lambda k: v[k].Synthetic[0]                                            # noqa: E731
    real = lambda k: " and ".join(v[k].fmt.format(tab.loc[v[k].Index, s][0]) for s in srcs[1:])  # noqa: E731
    lines += [
        "", "## What this means", "",
        f"* **Lows arrive too slowly, which is why synthetic lows were so easy to predict.** An hour before a low,",
        f"  synthetic glucose is already at {syn('pre_hypo_g60'):.0f} mg/dL and falls {abs(syn('pre_hypo_slope')):.2f} mg/dL per",
        f"  minute; real patients are at {real('pre_hypo_g60')} mg/dL and fall {real('pre_hypo_slope').replace('-', '')}",
        "  mg/dL per minute. This explains the drop in alert performance on real patients (76% to 35% of lows warned",
        "  30 minutes ahead, [real_model.md](real_model.md)).",
        f"* **Not too smooth, but jittery on top of slow.** We had written that simulated glucose is smoother than real.",
        f"  At the 15-minute scale it reverses direction more often than real CGM ({syn('reversal_rate'):.0f}% of steps",
        f"  vs {real('reversal_rate')}%): the sensor-noise model adds short wiggles, while large swings are rarer",
        f"  ({syn('delta_p95'):.0f} vs {real('delta_p95')} mg/dL) and overall variability is lower (CV {syn('cv'):.0f}%).",
        "* **Level, time in range and event rates are realistic.** Mean glucose, time in and above range, and the",
        "  rates of highs and lows all fall within the real cohorts' range.",
        "* **The daily rhythm differs.** Both real cohorts peak after breakfast (08:00-09:00); the simulator's largest rise",
        "  comes late in the evening. The hourly profile correlates weakly with the real cohorts' (they also differ",
        "  from each other). The simulator's meal and dosing times are a likely cause, not tested here.",
        "* **Lows without glucose-lowering drugs never happen in the simulator but do in real data** (ShanghaiT2DM:",
        "  patients on neither insulin nor a sulfonylurea have lows too; some may be sensor artefacts or other drugs).",
        "  Insulin users have fewer lows than sulfonylurea users in both the simulator and ShanghaiT2DM.", "",
        "**Next simulator changes this points to:** faster, steeper falls into lows; less short-term sensor jitter and",
        "more large physiological swings (meal size and timing variability); irregular daily schedules; and a small",
        "rate of lows in patients without insulin or sulfonylureas. Each can now be checked against this table.",
    ]
    lines += ["", "![Synthetic vs real CGM](figures/fig_simulator_realism.png)", "",
              "## Lows by drug class", "",
              "Mean per patient. CGMacros has no medication records.", "",
              "| Cohort | Group | Patients | Hypo episodes per day | Time below 70 (%) |", "|---|---|---|---|---|"]
    for r in drugs.itertuples():
        lines.append(f"| {r.source} | {r.drug_group} | {r.units} | {r.hypo_per_day:.2f} | {r.tbr:.2f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--shanghai-dir", default=str(ROOT / "data" / "external" / "shanghai"))
    ap.add_argument("--cgmacros-dir", default=str(ROOT / "data" / "external" / "cgmacros"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    dt = cfg["sampling"]["interval_minutes"]
    sources = {"Synthetic": synthetic_series(cfg)}
    for name, loader, d in (("ShanghaiT2DM", shanghai_series, args.shanghai_dir),
                            ("CGMacros T2D", cgmacros_series, args.cgmacros_dir)):
        try:
            sources[name] = loader(Path(d), dt)
        except FileNotFoundError as exc:
            print(f"skipping {name}: {exc}")
    if len(sources) < 2:
        raise SystemExit("no real cohort found; download ShanghaiT2DM and/or CGMacros first")
    per, trajs, profs = collect(sources)
    raw15 = {name: np.concatenate([np.diff(to_15min(s).to_numpy(float)) for s in ser])
             for name, (ser, _) in sources.items()}
    raw15 = {k: v[~np.isnan(v)] for k, v in raw15.items()}
    tab = summary_table(per, profs)
    drugs = drug_table(per)
    docs = resolve_path(cfg["paths"]["docs_dir"])
    plot(per, trajs, profs, raw15, docs / "figures" / "fig_simulator_realism.png")
    out = tab.drop(columns=["fmt"]).copy()
    for s in SOURCES:
        if s in out:
            out[s] = out[s].map(lambda v: None if not isinstance(v, tuple) else round(float(v[0]), 4))
    out.to_csv(docs / "simulator_realism.csv", index=False)
    write_report(tab, drugs, per, docs / "simulator_realism.md")
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    print(drugs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
