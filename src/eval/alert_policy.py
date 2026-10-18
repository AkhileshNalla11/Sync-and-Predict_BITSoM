"""Episode-level evaluation of the alert policy, the way a clinician judges alerts.

    python -m src.eval.alert_policy          # scores validation windows with the saved model; no retraining

Questions answered per target (hypo, nocturnal hypo, hyper):
  * what share of glucose EPISODES got an alert with >= 15 / >= 30 min warning,
  * how many alerts per patient-day (and per night) that costs, how many are false,
  * PPV per alert and the median lead time.

Protocol: every policy in a grid (threshold x hysteresis x refractory period x daily budget) is run on
VALIDATION patients; one operating point per target is picked there by a fixed rule (most episodes
caught with >= 30 min warning within an alert budget per patient-day). Only that point, and the old
window-level alerting for comparison, is then evaluated on the TEST patients (saved test predictions).

Writes docs/alert_policy.md, docs/alert_policy_sweep_val.csv, docs/alert_policy_test.csv and
docs/figures/fig_alert_tradeoff.png. SYNTHETIC DATA, NOT FOR CLINICAL USE.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.wearable_features import window_sizes
from src.eval.metrics import event_episodes, event_onsets, threshold_at_far
from src.twin.alerts import AlertPolicy, policy_alerts, policy_from_spec
from src.twin.state import TwinContext

# Sweep grid (validation only).
FARS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
REARM_FRACTIONS = (None, 1.0, 0.5, 0.25)                # None = no hysteresis; 1.0 = must drop below the threshold
REFRACTORY_MIN = (0, 60, 90, 120, 180, 240)
BUDGETS = (None, 4, 2)
ESCALATIONS = (None, 2.0, 3.0)                          # only meaningful with hysteresis
# Selection rule, fixed before looking at test: maximise episodes caught with >= SELECT_LEAD_MIN warning
# subject to alerts per patient-day <= the cap; ties (within TIE_PP points) go to fewer alerts per day.
SELECT_LEAD_MIN = 30
ALERT_CAP_PER_DAY = {"hypo": 1.0, "hyper": 2.0}
TIE_PP = 0.01
LEADS_MIN = (15, 30)
EPISODE_MIN = 15                                        # consensus: >= 15 min past the threshold, ended by >= 15 min clear
MODEL = "lgbm_cgm_wearables_ehr"
TARGET_LABEL = {"hypo": "Hypoglycaemia (< 70)", "hypo_noct": "Nocturnal hypoglycaemia", "hyper": "Hyperglycaemia (> 180)"}

# Same validated chart palette as the ablation figures.
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
POLICY, WINDOW = "#2a78d6", "#a3a29c"


# --------------------------------------------------------------------------- data
@dataclass
class Split:
    """One patient split: its prediction windows (time-ordered per patient), risk, and episode onsets."""

    windows: pd.DataFrame
    risk: dict[str, np.ndarray]
    episodes: dict[str, pd.DataFrame]          # target -> patient_idx, onset_idx, nocturnal
    stride_h: float
    horizon: int                               # samples
    dt: int                                    # minutes per sample

    @property
    def patient_days(self) -> float:
        return len(self.windows) * self.stride_h / 24


def _episode_table(onsets: list[np.ndarray], patients: np.ndarray, timestamps: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    from src.data.wearable_features import is_nocturnal

    rows = [(p, e) for p in patients for e in onsets[p]]
    df = pd.DataFrame(rows, columns=["patient_idx", "onset_idx"]).astype(int)
    thr = cfg["thresholds"]
    df["nocturnal"] = is_nocturnal(timestamps[df.onset_idx.to_numpy()], thr["nocturnal_start"], thr["nocturnal_end"])
    return df


def build_split(ctx: TwinContext, split: str, risk: dict[str, np.ndarray] | None = None,
                windows: pd.DataFrame | None = None, single_reading: bool = False) -> Split:
    cfg = ctx.cfg
    if windows is None:
        windows, risk = ctx.split_scores(split)
    order = np.lexsort((windows.anchor_idx.to_numpy(), windows.patient_idx.to_numpy()))
    windows = windows.iloc[order].reset_index(drop=True)
    risk = {t: np.asarray(v)[order] for t, v in risk.items()}
    _, horizon, stride = window_sizes(cfg)
    dt = cfg["sampling"]["interval_minutes"]
    patients = np.unique(windows.patient_idx.to_numpy())
    cgm = ctx.series[patients, :, 0]
    thr = cfg["thresholds"]
    n = EPISODE_MIN // dt
    eps = {}
    for t, level, below in (("hypo", thr["hypo_mgdl"], True), ("hyper", thr["hyper_mgdl"], False)):
        if single_reading:          # the ablation's lead-time definition: any reading past, after >= 30 min clear
            ons = event_onsets(cgm, level, below=below, clear_samples=30 // dt)
        else:
            ons = event_episodes(cgm, level, below=below, min_samples=n, clear_samples=n)
        full = [np.zeros(0, int)] * len(ctx.patient_ids)
        for p, o in zip(patients, ons):
            full[p] = o
        eps[t] = _episode_table(full, patients, ctx.timestamps, cfg)
    return Split(windows, risk, eps, stride * dt / 60, horizon, dt)


# --------------------------------------------------------------------------- metrics
def episode_leads(split: Split, alerts: np.ndarray, target: str) -> pd.DataFrame:
    """Per episode: eligible (any prediction in the 2 h before onset) and lead time of the earliest alert.

    An alert at anchor a warns of an episode with onset e if e - horizon <= a < e, i.e. the episode
    falls inside the alert's 2 h horizon. Lead = e - a for the earliest such alert (NaN if none).
    """
    w = split.windows
    key = w.patient_idx.to_numpy().astype(np.int64) * 10**7 + w.anchor_idx.to_numpy()
    akey = key[alerts]
    ep = split.episodes[target]
    e = ep.patient_idx.to_numpy().astype(np.int64) * 10**7 + ep.onset_idx.to_numpy()
    lo = e - split.horizon

    def first_in(keys: np.ndarray) -> np.ndarray:
        i = np.searchsorted(keys, lo, side="left")
        cand = keys[np.minimum(i, len(keys) - 1)] if len(keys) else np.full(len(e), np.iinfo(np.int64).max)
        return np.where((i < len(keys)) & (cand < e), cand, -1)

    first_pred = first_in(key)
    first_alert = first_in(akey)
    out = ep.copy()
    out["eligible"] = first_pred >= 0
    out["lead_min"] = np.where(first_alert >= 0, (e - first_alert) * split.dt, np.nan)
    return out


def per_patient(split: Split, alerts: np.ndarray, target: str) -> pd.DataFrame:
    """Per-patient counts behind every reported metric (summed for point estimates and bootstraps)."""
    w = split.windows
    y = w[target].to_numpy(bool)
    noct = w.nocturnal.to_numpy(bool)
    df = pd.DataFrame({"patient_idx": w.patient_idx.to_numpy(), "alerts": alerts, "true_alerts": alerts & y,
                       "noct_alerts": alerts & noct, "noct_windows": noct, "windows": 1})
    agg = df.groupby("patient_idx").sum()
    leads = episode_leads(split, alerts, target)
    leads = leads[leads.eligible]
    for name, sub in (("", leads), ("noct_", leads[leads.nocturnal])):
        g = sub.groupby("patient_idx")
        agg[f"{name}episodes"] = g.size()
        for lead in (0, *LEADS_MIN):
            agg[f"{name}caught_{lead}"] = g.lead_min.apply(lambda s, L=lead: int((s >= max(L, 1e-9)).sum()))
    return agg.fillna(0)


def summarise(pp: pd.DataFrame, split: Split, leads: pd.Series | None = None) -> dict[str, float]:
    s = pp.sum()
    days = s.windows * split.stride_h / 24
    nights = s.noct_windows * split.stride_h / 8                 # nocturnal block 22:00-06:00 = 8 h
    out = {"episodes": int(s.episodes), "noct_episodes": int(s.noct_episodes),
           "alerts_per_day": s.alerts / days, "false_alerts_per_day": (s.alerts - s.true_alerts) / days,
           "ppv": s.true_alerts / s.alerts if s.alerts else np.nan,
           "noct_alerts_per_night": s.noct_alerts / nights if nights else np.nan}
    for lead in (0, *LEADS_MIN):
        out[f"caught_{lead}"] = s[f"caught_{lead}"] / s.episodes if s.episodes else np.nan
        out[f"noct_caught_{lead}"] = s[f"noct_caught_{lead}"] / s.noct_episodes if s.noct_episodes else np.nan
    if leads is not None:
        out["median_lead_min"] = float(np.nanmedian(leads)) if np.isfinite(leads).any() else np.nan
    return out


def bootstrap(pp: pd.DataFrame, split: Split, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """95% CIs from resampling whole patients."""
    rng = np.random.default_rng(seed)
    vals = []
    arr = pp.to_numpy(float)
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        vals.append(summarise(pd.DataFrame(arr[idx], columns=pp.columns), split))
    df = pd.DataFrame(vals)
    return {c: (float(np.nanpercentile(df[c], 2.5)), float(np.nanpercentile(df[c], 97.5))) for c in df.columns}


def evaluate(split: Split, policy: AlertPolicy, target: str) -> tuple[dict[str, float], pd.DataFrame]:
    alerts = policy_alerts(split.windows, split.risk[target], policy, target)
    pp = per_patient(split, alerts, target)
    leads = episode_leads(split, alerts, target)
    return summarise(pp, split, leads[leads.eligible].lead_min), pp


# --------------------------------------------------------------------------- sweep and selection
def val_thresholds(val: Split) -> dict[str, dict[float, float]]:
    out = {}
    for t in ("hyper", "hypo"):
        onset = ~val.windows[f"in_event_{t}"].to_numpy(bool)
        y = val.windows[t].to_numpy(int)[onset]
        out[t] = {f: threshold_at_far(y, val.risk[t][onset], f) for f in FARS}
    return out


def sweep(val: Split, thresholds: dict[str, dict[float, float]]) -> pd.DataFrame:
    rows = []
    for t in ("hypo", "hyper"):
        for far in FARS:
            thr = thresholds[t][far]
            res, _ = evaluate(val, AlertPolicy.window_level(thr), t)
            rows.append({"target": t, "kind": "window", "false_alert_rate": far, "threshold": thr, "rearm_fraction": np.nan,
                         "refractory_min": 0, "daily_budget": np.nan, "escalation": np.nan, **res})
            for frac, refr, budget, esc in itertools.product(REARM_FRACTIONS, REFRACTORY_MIN, BUDGETS, ESCALATIONS):
                if frac is None and esc is not None:
                    continue
                spec = {"rearm_fraction": frac, "refractory_min": refr, "daily_budget": budget, "escalation": esc}
                res, _ = evaluate(val, policy_from_spec(spec, thr), t)
                rows.append({"target": t, "kind": "policy", "false_alert_rate": far, "threshold": thr,
                             "rearm_fraction": np.nan if frac is None else frac, "refractory_min": refr,
                             "daily_budget": np.nan if budget is None else budget,
                             "escalation": np.nan if esc is None else esc, **res})
        print(f"  swept {t}", flush=True)
    return pd.DataFrame(rows)


def select(sw: pd.DataFrame) -> dict[str, pd.Series]:
    """The validation operating point per target (see the rule at the top of this module)."""
    out = {}
    col = f"caught_{SELECT_LEAD_MIN}"
    for t, cap in ALERT_CAP_PER_DAY.items():
        c = sw[(sw.target == t) & (sw.kind == "policy") & (sw.alerts_per_day <= cap)]
        best = c[col].max()
        tied = c[c[col] >= best - TIE_PP]
        # prefer fewer alerts, then the simpler policy (fewer optional parts), then a shorter refractory period
        parts = tied[["rearm_fraction", "daily_budget", "escalation"]].notna().sum(axis=1)
        tied = tied.assign(_parts=parts).sort_values(["alerts_per_day", "_parts", "refractory_min"])
        out[t] = tied.iloc[0]
    return out


def components(sw: pd.DataFrame, chosen: dict[str, pd.Series]) -> pd.DataFrame:
    """Validation rows that add one policy component at a time, at each target's chosen threshold."""
    rows = []
    for t, c in chosen.items():
        d = sw[(sw.target == t) & (sw.false_alert_rate == c.false_alert_rate)]
        pol = d[d.kind == "policy"]
        nob, noe = pol.daily_budget.isna(), pol.escalation.isna()
        refr = c.refractory_min if c.refractory_min else 120
        steps = [("Every window above threshold (old)", d[d.kind == "window"]),
                 ("No alert while CGM already past threshold", pol[pol.rearm_fraction.isna() & (pol.refractory_min == 0) & nob]),
                 (f"+ refractory {refr / 60:g} h (reminder while risk stays high)",
                  pol[pol.rearm_fraction.isna() & (pol.refractory_min == refr) & nob]),
                 ("+ hysteresis: re-arm below 0.5 x threshold",
                  pol[(pol.rearm_fraction == 0.5) & (pol.refractory_min == refr) & nob & noe]),
                 ("+ hysteresis + escalation at 2x",
                  pol[(pol.rearm_fraction == 0.5) & (pol.refractory_min == refr) & nob & (pol.escalation == 2.0)])]
        for name, r in steps:
            if len(r):
                rows.append({"target": t, "step": name, **r.iloc[0][["caught_15", "caught_30", "alerts_per_day",
                                                                      "false_alerts_per_day", "ppv"]].to_dict()})
        rows.append({"target": t, "step": "**Chosen policy**", **c[["caught_15", "caught_30", "alerts_per_day",
                                                                    "false_alerts_per_day", "ppv"]].to_dict()})
    return pd.DataFrame(rows)


def spec_of(row: pd.Series) -> dict[str, Any]:
    return {"false_alert_rate": float(row.false_alert_rate),
            "rearm_fraction": None if pd.isna(row.rearm_fraction) else float(row.rearm_fraction),
            "refractory_min": int(row.refractory_min),
            "daily_budget": None if pd.isna(row.daily_budget) else int(row.daily_budget),
            "escalation": None if pd.isna(row.escalation) else float(row.escalation)}


# --------------------------------------------------------------------------- figure
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def frontier(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Upper-left Pareto frontier: most caught for a given alert rate."""
    d = df.dropna(subset=[x, y])
    d = d[d[y] > 0].sort_values([x, y], ascending=[True, False])
    keep, best = [], -np.inf
    for i, r in d.iterrows():
        if r[y] > best:
            keep.append(i)
            best = r[y]
    return d.loc[keep]


def plot_tradeoff(sw: pd.DataFrame, chosen: dict[str, pd.Series], test: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    panels = [("hypo", "", "alerts_per_day", "Hypo alerts per patient-day"),
              ("hypo", "noct_", "noct_alerts_per_night", "Hypo alerts per patient-night (22:00-06:00)"),
              ("hyper", "", "alerts_per_day", "Hyper alerts per patient-day")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), facecolor=SURFACE)
    for ax, (t, pre, xcol, xlabel) in zip(axes, panels):
        ycol = f"{pre}caught_{SELECT_LEAD_MIN}"
        d = sw[sw.target == t]
        pol = frontier(d[d.kind == "policy"], xcol, ycol)
        win = d[d.kind == "window"].sort_values(xcol)
        ax.plot(win[xcol], win[ycol] * 100, color=WINDOW, lw=2, marker="o", ms=4, label="Every window above threshold (old)")
        ax.plot(pol[xcol], pol[ycol] * 100, color=POLICY, lw=2, label="Alert policy, best per alert rate (validation)")
        c = chosen[t]
        ax.scatter([c[xcol]], [c[ycol] * 100], s=80, facecolor=SURFACE, edgecolor=POLICY, linewidth=2, zorder=4)
        r = test[(test.target == t) & (test.kind == "policy")].iloc[0]
        ax.scatter([r[xcol]], [r[ycol] * 100], s=80, color=POLICY, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.text(0.97, 0.05, f"Test: {r[ycol]:.0%} warned >= {SELECT_LEAD_MIN} min ahead\n"
                f"at {r[xcol]:.2f} alerts", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
        if pre == "":
            ax.axvline(ALERT_CAP_PER_DAY[t], color=INK_3, ls="--", lw=1)
            ax.text(ALERT_CAP_PER_DAY[t], 2, " cap", fontsize=7, color=INK_2)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(0, 102)
        ax.set_xlabel(xlabel + " (log scale)", fontsize=8, color=INK_2)
        ax.set_title(TARGET_LABEL[t + ("_noct" if pre else "")], fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel(f"% of episodes warned >= {SELECT_LEAD_MIN} min ahead", fontsize=8, color=INK_2)
    handles = [Line2D([], [], color=WINDOW, lw=2, marker="o", ms=4, label="Every window above threshold (old), validation"),
               Line2D([], [], color=POLICY, lw=2, label="Alert policy frontier, validation"),
               Line2D([], [], ls="", marker="o", ms=8, mfc=SURFACE, mec=POLICY, mew=2, label="Chosen point, validation"),
               Line2D([], [], ls="", marker="o", ms=8, color=POLICY, label="Chosen point, test")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Episodes caught vs alert burden. SYNTHETIC DATA: simulated lows are slow-onset and easy to anticipate.",
                 fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- report
def _pct_ci(r: pd.Series, c: str) -> str:
    return f"{r[c]:.0%} ({r[c + '_lo']:.0%}-{r[c + '_hi']:.0%})" if np.isfinite(r[c]) else "n/a"


def _num_ci(r: pd.Series, c: str, fmt: str = ".2f") -> str:
    return f"{r[c]:{fmt}} ({r[c + '_lo']:{fmt}}-{r[c + '_hi']:{fmt}})"


def write_markdown(sw: pd.DataFrame, chosen: dict[str, pd.Series], test: pd.DataFrame, path: Path) -> None:
    tp = {(r.target, r.kind): r for _, r in test.iterrows()}
    h, n = tp[("hypo", "policy")], tp[("hyper", "policy")]
    hw, nw = tp[("hypo", "window")], tp[("hyper", "window")]
    comp = components(sw, chosen)
    hc = comp[comp.target == "hypo"].set_index("step")
    refr_row = next(s for s in hc.index if s.startswith("+ refractory"))
    hyst_row = "+ hysteresis: re-arm below 0.5 x threshold"
    lines = [
        "# Alert policy: one alert per risk episode", "",
        "**SYNTHETIC DATA. Not clinically validated.** The numbers describe how the alert logic behaves on simulated",
        "patients whose lows are slow and easy to see coming. They are not an estimate of real-world performance.", "",
        "## Headline (60 test patients)", "",
        f"* **Hypoglycaemia:** the policy warns of **{h.caught_30:.0%} of hypo episodes at least 30 minutes before onset**",
        f"  ({h.caught_15:.0%} with >= 15 min) with **{h.alerts_per_day:.2f} hypo alerts per patient-day**, of which",
        f"  {h.false_alerts_per_day:.2f} are false (PPV {h.ppv:.0%} per alert; median lead {h.median_lead_min:.0f} min).",
        f"  Nocturnal lows: {h.noct_caught_30:.0%} warned >= 30 min ahead, {h.noct_alerts_per_night:.2f} hypo alerts per night.",
        f"* **Hyperglycaemia:** {n.caught_30:.0%} of hyper episodes warned >= 30 min ahead with {n.alerts_per_day:.2f} alerts",
        f"  per patient-day (PPV {n.ppv:.0%}).",
        f"* **Before** (every 30-minute window at or above the 5%-false-alert-rate threshold counted as an alert):",
        f"  {hw.alerts_per_day:.1f} hypo and {nw.alerts_per_day:.1f} hyper alerting windows per patient-day, with the",
        "  same alarm repeating every 30 min while risk stays high.",
        f"* **The price:** for {hw.alerts_per_day / h.alerts_per_day:.0f}x fewer hypo alerts the policy gives up "
        f"{(hw.caught_30 - h.caught_30) * 100:.0f} points of",
        f"  episodes caught >= 30 min ahead ({hw.caught_30:.0%} -> {h.caught_30:.0%}) and median lead time falls from "
        f"{hw.median_lead_min:.0f} to {h.median_lead_min:.0f} min.",
        "  Where to sit on that curve is a clinical choice (figure below).",
    ]
    for t, r in (("hypo", h), ("hyper", n)):
        if r.alerts_per_day > ALERT_CAP_PER_DAY[t]:
            lines.append(f"* On test the {t} policy raises {r.alerts_per_day:.2f} alerts per patient-day, above the "
                         f"{ALERT_CAP_PER_DAY[t]:g} cap it met on validation ({chosen[t].alerts_per_day:.2f}).")
    lines += ["", "## Method", "",
              "**Policy** (`src/twin/alerts.py`, one per target). An alert fires when the predicted 2-hour risk reaches the",
              "threshold. After that, the policy stays quiet until the **refractory period** has passed. Optional parts:",
              "* **Hysteresis:** re-arm only after risk has fallen below a lower level.",
              "* **Escalation:** with hysteresis, alert again if risk rises to k x its level at the last alert.",
              "* **Daily budget:** at most N alerts per patient per calendar day.",
              "",
              "No predictive alert is raised while the CGM already reads past the threshold: the CGM's own threshold alarm",
              "covers that case. All parts are implemented and tested; validation decides which to use.", "",
              "**Episodes.** Consensus-style CGM episodes: >= 15 min (3 readings) past the threshold, ended by >= 15 min",
              "clear of it. Nocturnal = onset 22:00-06:00. An episode is *eligible* if any prediction was made in the 2 h",
              "before onset (episodes inside data gaps cannot be warned of and are excluded).", "",
              "**Metrics.**",
              "* *Caught with >= L min warning:* an alert fired between 2 h and L min before onset (the alert's 2-hour",
              "  horizon contains the onset).",
              "* *Lead time:* onset minus the earliest such alert. Predictions run every 30 min, so lead time is capped",
              "  at 120 min.",
              "* *PPV per alert:* share of alerts followed by any reading past the threshold within 2 h (the model's label).",
              "* *False alerts per patient-day:* the remaining alerts, per 24 h of monitored time (30 min per prediction).",
              "",
              "**Choosing the operating point on validation patients only.** Every combination below was run on the",
              f"{sw.attrs.get('n_val_patients', 45)} validation patients:",
              f"* threshold: false-alert rate on validation onset windows of {', '.join(f'{f:g}' for f in FARS)};",
              "* re-arm level: none, or 1.0 / 0.5 / 0.25 x threshold;",
              f"* refractory period: {', '.join(f'{r / 60:g}' for r in REFRACTORY_MIN)} h;",
              "* daily budget: none, 4 or 2; escalation (with hysteresis only): none, 2x or 3x.",
              "",
              f"The rule was fixed in advance: maximise episodes caught with >= {SELECT_LEAD_MIN} min warning, subject to at most",
              f"{ALERT_CAP_PER_DAY['hypo']:g} hypo or {ALERT_CAP_PER_DAY['hyper']:g} hyper alerts per patient-day. Within",
              f"{TIE_PP * 100:g} point of the best, take the fewest alerts, then the simplest policy. The chosen points are in",
              "`config.yaml` (`twin.alert_policy`), which the dashboard uses. Test patients played no part.", "",
              "| Target | Threshold (val FAR) | Refractory | Re-arm below | Escalation | Daily budget | Val: caught >= 30 min | Val: alerts / day |",
              "|---|---|---|---|---|---|---|---|"]
    for t in ("hypo", "hyper"):
        c = chosen[t]
        rearm = "none" if pd.isna(c.rearm_fraction) else f"{c.rearm_fraction:g} x threshold"
        esc = "none" if pd.isna(c.escalation) else f"{c.escalation:g}x"
        lines.append(f"| {TARGET_LABEL[t]} | {c.threshold:.1%} ({c.false_alert_rate * 100:g}%) | {c.refractory_min / 60:g} h | "
                     f"{rearm} | {esc} | {'none' if pd.isna(c.daily_budget) else int(c.daily_budget)} | "
                     f"{c.caught_30:.0%} | {c.alerts_per_day:.2f} |")
    lines += ["", "### What each part does (validation patients, chosen threshold)", "",
              f"The refractory period does most of the work: for hypo it cuts alerts from "
              f"{hc.loc['Every window above threshold (old)'].alerts_per_day:.2f} to {hc.loc[refr_row].alerts_per_day:.2f}",
              f"per patient-day while episodes caught >= 30 min fall from "
              f"{hc.loc['Every window above threshold (old)'].caught_30:.0%} to {hc.loc[refr_row].caught_30:.0%}.",
              f"**Hysteresis hurts here** (hypo caught >= 30 min: {hc.loc[hyst_row].caught_30:.0%}). Simulated lows are slow",
              "drifts: risk crosses the threshold, often more than 2 h before onset, and then stays high. A policy that",
              "waits for risk to fall before re-arming stays silent exactly when the low arrives. A refractory period",
              "without hysteresis acts as a reminder at most every 1.5 h while risk stays high, which is what catches",
              "these lows. Escalation recovers only part of the loss. The daily budget only removes alerts once the",
              "cap is tight enough to drop real warnings.", "",
              "| Target | Policy | Caught >= 15 min | Caught >= 30 min | Alerts / patient-day | False alerts / patient-day | PPV |",
              "|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        lines.append(f"| {r.target} | {r.step} | {r.caught_15:.0%} | {r.caught_30:.0%} | {r.alerts_per_day:.2f} | "
                     f"{r.false_alerts_per_day:.2f} | {r.ppv:.0%} |")
    lines += ["", "PPV of window-level alerting looks high partly because it includes windows where the CGM is already",
              "past the threshold, which are trivially positive.", "",
              "## Test results (60 patients, 95% patient-bootstrap CIs)", "",
              "*Old* = every 30-min window at or above the 5%-FAR threshold counts as an alert (previous dashboard).",
              "*Same threshold, every window* isolates the policy from the threshold change. *Policy* = chosen above.", "",
              "| Target | Alerting | Episodes | Caught (any warning) | Caught >= 15 min | Caught >= 30 min | Median lead (min) |"
              " Alerts / patient-day | False alerts / patient-day | PPV per alert |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in test.iterrows():
        lines.append(f"| {TARGET_LABEL[r.target]} | {r.kind_label} | {r.episodes} | {_pct_ci(r, 'caught_0')} | "
                     f"{_pct_ci(r, 'caught_15')} | {_pct_ci(r, 'caught_30')} | {r.median_lead_min:.0f} | "
                     f"{_num_ci(r, 'alerts_per_day')} | {_num_ci(r, 'false_alerts_per_day')} | {_pct_ci(r, 'ppv')} |")
    lines += ["", "**Nocturnal hypoglycaemia (test).** Onset 22:00-06:00; alerts counted per 8-hour night.", "",
              "| Alerting | Nocturnal episodes | Caught >= 15 min | Caught >= 30 min | Hypo alerts per night |", "|---|---|---|---|---|"]
    for _, r in test[test.target == "hypo"].iterrows():
        lines.append(f"| {r.kind_label} | {r.noct_episodes} | {_pct_ci(r, 'noct_caught_15')} | {_pct_ci(r, 'noct_caught_30')} | "
                     f"{_num_ci(r, 'noct_alerts_per_night')} |")
    sr = test.attrs.get("single_reading")
    if sr is not None:
        lines += ["", "*Sensitivity to the episode definition.* Counting any single reading < 70 as an onset (after >= 30 min",
                  f"clear; the definition in the ablation lead-time table) gives {sr['episodes']} hypo episodes. The policy",
                  f"catches {sr['caught_30']:.0%} of them with >= 30 min warning ({sr['caught_15']:.0%} with >= 15 min)."]
    lines += ["", "## Trade-off: episodes caught vs alert burden", "",
              "Lines are computed on validation patients. Grey: the old window-level alerting at each threshold. Blue:",
              "for each alert rate, the best policy in the grid. Hollow dot = chosen point on validation; filled dot = the",
              "same policy on test. Log x-axis.", "",
              "![Alert trade-off](figures/fig_alert_tradeoff.png)", "",
              "## Caveats", "",
              "* **Simulated lows are slow-onset.** In the 2 h before a simulated hypo onset, glucose is typically already",
              "  drifting toward 70 mg/dL (median lowest reading 70 mg/dL, see [ablation_results.md](ablation_results.md)),",
              "  so warnings 30+ min ahead are easy here. Real lows after exercise or mistimed insulin can fall much",
              "  faster; expect fewer episodes caught and shorter lead times on real data. The same slow drifts are",
              "  why hysteresis failed; on real data with faster dynamics it may be worth re-testing.",
              "* **No human in the loop.** Alerts are scored as if nobody acted on them; in practice a treated alert",
              "  prevents the episode and would then look like a false alert.",
              "* **Coarse lead time.** Predictions run every 30 min with a 2-hour horizon. Lead times are capped at",
              "  120 min, and an alert more than 2 h before onset does not count as a warning.",
              "* **PPV depends on prevalence.** Hypo episodes are rare, so the false alerts per day are the number a",
              "  patient actually lives with.",
              f"* **Small cohorts.** {sw.attrs.get('n_val_patients', 45)} validation and 60 test patients; the CIs resample",
              "  whole patients and are wide for nocturnal hypo.",
              "* **The alert cap is a clinical judgement**, not a fitted quantity; a different cap gives a different",
              "  point on the trade-off curve. Full validation sweep: [alert_policy_sweep_val.csv](alert_policy_sweep_val.csv).",
              "  Test rows: [alert_policy_test.csv](alert_policy_test.csv).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- main
def load_test(ctx: TwinContext) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Test windows + the deployed model's saved test predictions."""
    res_dir = Path(resolve_path(ctx.cfg["paths"]["processed_dir"])) / "results"
    preds = pd.read_csv(res_dir / "baselines_test_predictions.csv.gz")
    w = ctx.windows.iloc[preds.window_id.to_numpy()].rename_axis("window_id").reset_index()
    if not (w.split == "test").all():
        raise ValueError("saved test predictions do not line up with the test windows; re-run the pipeline")
    return w, {t: preds[f"{MODEL}_{t}"].to_numpy() for t in ("hyper", "hypo")}


def run(cfg: dict[str, Any], sweep_csv: str | None = None) -> dict[str, Any]:
    if cfg["twin"]["model"] != MODEL:
        raise ValueError(f"alert policy evaluation expects twin.model = {MODEL}")
    ctx = TwinContext.load(cfg)
    print("  scoring validation windows ...", flush=True)
    val = build_split(ctx, "val")
    tw, trisk = load_test(ctx)
    test = build_split(ctx, "test", trisk, tw)
    # sanity: the saved test predictions are the deployed model's
    w0, p0 = ctx.split_scores("test")
    if not np.allclose(pd.Series(p0["hypo"], index=w0.index).loc[tw.window_id].to_numpy(), trisk["hypo"], atol=1e-6):
        raise ValueError("saved test predictions differ from the deployed model's; re-run the pipeline")

    thr = val_thresholds(val)
    if sweep_csv:                                           # reuse a saved validation sweep (same grid)
        sw = pd.read_csv(sweep_csv)
    else:
        print("  sweeping policies on validation patients ...", flush=True)
        sw = sweep(val, thr)
    sw.attrs["n_val_patients"] = int(val.windows.patient_id.nunique())
    chosen = select(sw)

    deployed_far = 0.05
    rows = []
    for t in ("hypo", "hyper"):
        test_thr = thr[t]                                   # thresholds always come from validation
        far = float(chosen[t].false_alert_rate)
        for kind, label, pol in (
                ("window", "Old: every window, 5% FAR", AlertPolicy.window_level(test_thr[deployed_far])),
                ("window_same", f"Same threshold ({far * 100:g}% FAR), every window", AlertPolicy.window_level(test_thr[far])),
                ("policy", "**Policy (chosen on validation)**", policy_from_spec(spec_of(chosen[t]), test_thr[far]))):
            res, pp = evaluate(test, pol, t)
            ci = bootstrap(pp, test, cfg["eval"]["n_bootstrap"], cfg["seed"])
            rows.append({"target": t, "kind": kind, "kind_label": label, "policy": pol.describe(), **res,
                         **{f"{k}_lo": v[0] for k, v in ci.items()}, **{f"{k}_hi": v[1] for k, v in ci.items()}})
    test_df = pd.DataFrame(rows)
    single = build_split(ctx, "test", trisk, tw, single_reading=True)
    hp = policy_from_spec(spec_of(chosen["hypo"]), thr["hypo"][float(chosen["hypo"].false_alert_rate)])
    test_df.attrs["single_reading"], _ = evaluate(single, hp, "hypo")

    docs = resolve_path(cfg["paths"]["docs_dir"])
    figs = resolve_path(Path(cfg["paths"]["docs_dir"]) / "figures")
    sw.to_csv(docs / "alert_policy_sweep_val.csv", index=False, float_format="%.6g")
    test_df.to_csv(docs / "alert_policy_test.csv", index=False, float_format="%.6g")
    import matplotlib
    matplotlib.use("Agg")
    plot_tradeoff(sw, chosen, test_df, figs / "fig_alert_tradeoff.png")
    write_markdown(sw, chosen, test_df, docs / "alert_policy.md")

    for t in ("hypo", "hyper"):
        want, have = spec_of(chosen[t]), {k: cfg["twin"]["alert_policy"][t].get(k) for k in spec_of(chosen[t])}
        if want != have:
            print(f"  NOTE: config.yaml twin.alert_policy.{t} = {have}, but validation selects {want}; update config.yaml")
    return {"sweep": sw, "chosen": chosen, "test": test_df}


def main() -> None:
    ap = argparse.ArgumentParser(description="Episode-level evaluation of the alert policy")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sweep-csv", default=None, help="reuse a saved validation sweep instead of re-running it")
    args = ap.parse_args()
    out = run(load_config(args.config), args.sweep_csv)
    cols = ["target", "kind", "policy", "episodes", "caught_0", "caught_15", "caught_30", "median_lead_min",
            "alerts_per_day", "false_alerts_per_day", "ppv", "noct_caught_30", "noct_alerts_per_night"]
    print(out["test"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
