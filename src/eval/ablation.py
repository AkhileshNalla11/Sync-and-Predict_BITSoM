"""Stage 5 ablation: does each data stream, and above all the EHR, add predictive value?

    python -m src.eval.ablation     # reads saved test predictions; no retraining

Ladder, on the same test patients:
    persistence  ->  LightGBM CGM  ->  + wearables  ->  + EHR            (tabular family)
                     GRU sequence-only (CGM + wearables)  ->  + EHR branch (neural family)

The EHR question is answered by PAIRED patient-bootstrap differences in AUPRC between otherwise
identical models (same features / architecture with vs without the static stream).

Writes docs/ablation_results.{md,csv}, docs/ablation_ehr_value.csv, docs/ablation_calibration.csv,
docs/ablation_lead_time.csv and docs/figures/*.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.build_dataset import load_processed
from src.data.wearable_features import window_sizes
from src.eval.metrics import (DECILES, TAIL_QUANTILES, auprc, calibration_summary, calibration_table, evaluate_targets,
                              evaluation_subsets, event_onsets, lead_time_stats, paired_bootstrap_diff, threshold_at_far)

# model key -> (display label, uses EHR?)
MODELS: dict[str, tuple[str, bool]] = {
    "persistence": ("Persistence (trend extrapolation)", False),
    "lgbm_cgm": ("LightGBM: CGM only", False),
    "lgbm_cgm_wearables": ("LightGBM: CGM + wearables", False),
    "lgbm_cgm_wearables_ehr": ("LightGBM: CGM + wearables + EHR", True),
    "seqonly_gru": ("GRU: sequence only", False),
    "fusion_gru": ("GRU fusion: sequence + EHR", True),
    "fusion_cnn": ("1D-CNN fusion: sequence + EHR", True),
}
# (model A, model B, question)
COMPARISONS = [
    ("lgbm_cgm_wearables_ehr", "lgbm_cgm_wearables", "LightGBM: adding EHR"),
    ("fusion_gru", "seqonly_gru", "GRU: adding EHR branch"),
    ("lgbm_cgm_wearables", "lgbm_cgm", "LightGBM: adding wearables"),
    ("lgbm_cgm", "persistence", "LightGBM CGM vs persistence"),
    ("fusion_gru", "lgbm_cgm_wearables_ehr", "GRU fusion vs LightGBM + EHR"),
]
HEADLINE = [("hyper", "onset"), ("hypo", "onset"), ("hypo", "nocturnal_onset")]
TARGET_LABEL = {("hyper", "onset"): "Hyperglycaemia (> 180)", ("hypo", "onset"): "Hypoglycaemia (< 70)",
                ("hypo", "nocturnal_onset"): "Nocturnal hypoglycaemia"}

# Chart palette (validated with the dataviz validator, light surface): color = "uses EHR?".
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
NO_EHR, WITH_EHR, REFERENCE = "#2a78d6", "#eb6834", "#a3a29c"


def model_color(key: str) -> str:
    return REFERENCE if key == "persistence" else (WITH_EHR if MODELS[key][1] else NO_EHR)


# --------------------------------------------------------------------------- data
def load_predictions(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Test windows + per-model, per-target scores aligned row by row."""
    ds = load_processed(cfg)
    res_dir = Path(resolve_path(cfg["paths"]["processed_dir"])) / "results"
    base = pd.read_csv(res_dir / "baselines_test_predictions.csv.gz")
    fusion = pd.read_csv(res_dir / "fusion_test_predictions.csv.gz")
    if not (base.window_id.to_numpy() == fusion.window_id.to_numpy()).all():
        raise ValueError("baseline and fusion predictions are for different test windows; re-run the pipeline")
    preds = pd.concat([base, fusion.drop(columns="window_id")], axis=1)
    test_w = ds["windows"].iloc[preds.window_id.to_numpy()].reset_index(drop=True)
    scores = {m: {t: preds[f"{m}_{t}"].to_numpy() for t in ("hyper", "hypo")}
              for m in MODELS if f"{m}_hyper" in preds.columns}
    return test_w, preds, scores, ds


# --------------------------------------------------------------------------- analyses
def discrimination(test_w: pd.DataFrame, scores: dict, cfg: dict) -> pd.DataFrame:
    rows = []
    for m, s in scores.items():
        rows += [{"model": m, "label": MODELS[m][0], "uses_ehr": MODELS[m][1], **r} for r in evaluate_targets(test_w, s, cfg)]
    return pd.DataFrame(rows)


def ehr_value(test_w: pd.DataFrame, scores: dict, cfg: dict) -> pd.DataFrame:
    masks = {(t, s): m for t, s, m in evaluation_subsets(test_w)}
    rows = []
    for a, b, question in COMPARISONS:
        if a not in scores or b not in scores:
            continue
        for target, subset in HEADLINE:
            m = masks[(target, subset)]
            d = paired_bootstrap_diff(test_w[target].to_numpy(int)[m], scores[a][target][m], scores[b][target][m],
                                      test_w.patient_id.to_numpy()[m], auprc, cfg["eval"]["n_bootstrap"], cfg["seed"])
            rows.append({"comparison": question, "model_a": a, "model_b": b, "target": target, "subset": subset,
                         "auprc_a": auprc(test_w[target].to_numpy(int)[m], scores[a][target][m]),
                         "auprc_b": auprc(test_w[target].to_numpy(int)[m], scores[b][target][m]), **d})
    return pd.DataFrame(rows)


def calibration(test_w: pd.DataFrame, scores: dict) -> tuple[pd.DataFrame, dict]:
    """Probability models only (persistence scores are not probabilities). All windows, as deployed."""
    rows, curves = [], {}
    for m, s in scores.items():
        if m == "persistence":
            continue
        for t in ("hyper", "hypo"):
            y = test_w[t].to_numpy(int)
            q = DECILES if t == "hyper" else TAIL_QUANTILES
            rows.append({"model": m, "target": t, **calibration_summary(y, s[t], q)})
            curves[(m, t)] = calibration_table(y, s[t], q)
    return pd.DataFrame(rows), curves


def lead_times(test_w: pd.DataFrame, scores: dict, ds: dict, cfg: dict) -> pd.DataFrame:
    """Detection rate / lead time at each configured false-alert operating point of the onset subset."""
    _, horizon, _ = window_sizes(cfg)
    dt = cfg["sampling"]["interval_minutes"]
    thr = cfg["thresholds"]
    cgm = ds["series"][..., 0]
    clear = 30 // dt
    onsets = {"hyper": event_onsets(cgm, thr["hyper_mgdl"], below=False, clear_samples=clear),
              "hypo": event_onsets(cgm, thr["hypo_mgdl"], below=True, clear_samples=clear)}
    masks = {(t, s): m for t, s, m in evaluation_subsets(test_w)}
    # How "visible" are events beforehand? Median CGM extreme in the 2 h before each test-patient onset.
    test_p = set(test_w.patient_idx)
    prior = {}
    for t, fn in (("hyper", np.nanmax), ("hypo", np.nanmin)):
        vals = [fn(cgm[p, max(0, e - horizon):e]) for p, ons in enumerate(onsets[t]) if p in test_p for e in ons if e > 0]
        prior[t] = float(np.nanmedian(vals)) if vals else np.nan
    rows = []
    for far in cfg["eval"]["false_alert_rates"]:
        for m, s in scores.items():
            for t in ("hyper", "hypo"):
                on = masks[(t, "onset")]
                threshold = threshold_at_far(test_w[t].to_numpy(int)[on], s[t][on], far)
                stats = lead_time_stats(onsets[t], test_w, s[t], threshold, f"in_event_{t}", horizon, dt)
                rows.append({"model": m, "target": t, "false_alert_rate": far, **stats,
                             "median_prior_2h_extreme_mgdl": prior[t]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_auprc(disc: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    models = [m for m in MODELS if m in set(disc.model)][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (t, sub) in zip(axes, HEADLINE):
        d = disc[(disc.target == t) & (disc.subset == sub)].set_index("model").loc[models]
        y = np.arange(len(models))
        err = np.vstack([d.auprc - d.auprc_lo, d.auprc_hi - d.auprc])
        ax.barh(y, d.auprc, height=0.62, color=[model_color(m) for m in models], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(d.auprc, y, xerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
        for yi, v, hi in zip(y, d.auprc, d.auprc_hi):
            ax.text(hi + 0.015, yi, f"{v:.2f}", va="center", ha="left", fontsize=8, color=INK)
        ax.axvline(d.prevalence.iloc[0], color=INK_3, ls="--", lw=1)
        ax.text(d.prevalence.iloc[0] + 0.01, len(models) - 0.45, "chance", fontsize=7, color=INK_2)
        ax.set_xlim(0, 1)
        ax.set_title(TARGET_LABEL[(t, sub)], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("AUPRC, onset windows (95% CI)", fontsize=8, color=INK_2)
        _style(ax)
    axes[0].set_yticks(np.arange(len(models)), [MODELS[m][0] for m in models], fontsize=8, color=INK)
    fig.legend(handles=[Patch(color=WITH_EHR, label="uses EHR"), Patch(color=NO_EHR, label="no EHR"),
                        Patch(color=REFERENCE, label="reference")], loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Ablation on test patients. SYNTHETIC DATA: the EHR signal is built into the simulator.",
                 fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_ehr_gain(ev: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    q = ["LightGBM: adding EHR", "GRU: adding EHR branch", "LightGBM: adding wearables"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0), sharey=True, facecolor=SURFACE)
    for ax, (t, sub) in zip(axes, HEADLINE):
        d = ev[(ev.target == t) & (ev.subset == sub)].set_index("comparison").loc[q]
        y = np.arange(len(q))[::-1]
        ax.axvline(0, color=INK_3, lw=1)
        colors = [WITH_EHR if "EHR" in c else NO_EHR for c in q]
        ax.hlines(y, d.diff_lo, d.diff_hi, color=colors, lw=2)
        ax.scatter(d["diff"], y, s=64, color=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.text(r.diff_hi + 0.004, yi, f"{r['diff']:+.3f}", va="center", fontsize=8, color=INK)
        ax.set_title(TARGET_LABEL[(t, sub)], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("AUPRC difference, paired patient bootstrap 95% CI", fontsize=8, color=INK_2)
        _style(ax)
    axes[0].set_yticks(np.arange(len(q))[::-1], q, fontsize=8, color=INK)
    lo, hi = ev[ev.comparison.isin(q)].diff_lo.min(), ev[ev.comparison.isin(q)].diff_hi.max()
    for ax in axes:
        ax.set_xlim(min(lo, -0.02) - 0.01, hi + 0.05)
    fig.suptitle("Value added by each stream (right of zero = helps). SYNTHETIC DATA.", fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_pr_curves(test_w: pd.DataFrame, scores: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    show = [m for m in ("persistence", "lgbm_cgm", "lgbm_cgm_wearables_ehr", "fusion_gru") if m in scores]
    style = {"persistence": (REFERENCE, ":"), "lgbm_cgm": (NO_EHR, "-"), "lgbm_cgm_wearables_ehr": (WITH_EHR, "-"),
             "fusion_gru": (WITH_EHR, "--")}
    masks = {(t, s): m for t, s, m in evaluation_subsets(test_w)}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), facecolor=SURFACE)
    for ax, key in zip(axes, HEADLINE):
        m = masks[key]
        y = test_w[key[0]].to_numpy(int)[m]
        for model in show:
            prec, rec, _ = precision_recall_curve(y, scores[model][key[0]][m])
            c, ls = style[model]
            ax.plot(rec, prec, color=c, ls=ls, lw=2, label=MODELS[model][0])
        ax.axhline(y.mean(), color=INK_3, lw=1, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(TARGET_LABEL[key], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Recall (sensitivity)", fontsize=8, color=INK_2)
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("Precision (PPV)", fontsize=8, color=INK_2)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Precision-recall on onset windows, test patients (dashed gray = prevalence). SYNTHETIC DATA.",
                 fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_calibration(curves: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    show = [m for m in ("lgbm_cgm_wearables_ehr", "fusion_gru") if (m, "hyper") in curves]
    style = {"lgbm_cgm_wearables_ehr": "-", "fusion_gru": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, t in zip(axes, ("hyper", "hypo")):
        lim = 1.0 if t == "hyper" else max(max(curves[(m, t)].mean_predicted.max(), curves[(m, t)].observed_rate.max()) for m in show) * 1.05
        ax.plot([0, lim], [0, lim], color=INK_3, lw=1)
        for m in show:
            c = curves[(m, t)]
            ax.plot(c.mean_predicted, c.observed_rate, color=WITH_EHR, ls=style[m], lw=2, marker="o", ms=5,
                    markeredgecolor=SURFACE, label=MODELS[m][0])
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title("Hyperglycaemia" if t == "hyper" else "Hypoglycaemia", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Mean predicted probability (" + ("decile bins" if t == "hyper" else "upper-tail quantile bins") + ")",
                      fontsize=8, color=INK_2)
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("Observed event rate", fontsize=8, color=INK_2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Calibration on all test windows (diagonal = perfect). SYNTHETIC DATA.", fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- report
def _ci(r: pd.Series, m: str) -> str:
    return f"{r[m]:.3f} ({r[m + '_lo']:.3f}-{r[m + '_hi']:.3f})"


def write_markdown(disc: pd.DataFrame, ev: pd.DataFrame, cal: pd.DataFrame, lead: pd.DataFrame, path: Path) -> None:
    models = [m for m in MODELS if m in set(disc.model)]
    d = disc.set_index(["model", "target", "subset"])
    prev = {k: disc[(disc.target == k[0]) & (disc.subset == k[1])].prevalence.iloc[0] for k in HEADLINE}
    lines = [
        "# Ablation results", "",
        "**SYNTHETIC DATA. Not clinically validated.** The EHR traits drive the simulated glucose dynamics *by design*,",
        "so the EHR gains below show that the pipeline *can* use static context; they do not estimate the value of",
        "EHR data in real patients. Test set: 60 patients, never seen in training or model selection.", "",
        "*Onset* windows exclude predictions made while the patient is already past the threshold (those are",
        "trivially positive). AUPRC is the headline metric (chance = prevalence); 95% CIs resample whole patients.", "",
        "## Headline: AUPRC on onset windows", "",
        "| Model | Uses EHR | Hyper > 180 | Hypo < 70 | Nocturnal hypo | Hypo AUROC | Hypo sens @ 10% FAR |",
        "|---|---|---|---|---|---|---|",
        f"| *chance (prevalence)* | | {prev[HEADLINE[0]]:.3f} | {prev[HEADLINE[1]]:.3f} | {prev[HEADLINE[2]]:.3f} | 0.5 | 0.10 |",
    ]
    for m in models:
        cells = [_ci(d.loc[(m, t, s)], "auprc") for t, s in HEADLINE]
        h = d.loc[(m, "hypo", "onset")]
        lines.append(f"| {MODELS[m][0]} | {'yes' if MODELS[m][1] else 'no'} | {' | '.join(cells)} | {h.auroc:.3f} | {h.sens_at_far10:.3f} |")
    lines += ["", "![AUPRC ablation](figures/fig_auprc_ablation.png)", "",
              "## Does the EHR add value? (paired AUPRC differences)", "",
              "A minus B on identical patient resamples. *P(A better)* is the share of 200 resamples where A wins.", "",
              "| Comparison | Target | AUPRC A | AUPRC B | Difference (95% CI) | P(A better) |", "|---|---|---|---|---|---|"]
    for _, r in ev.iterrows():
        lines.append(f"| {r.comparison} | {TARGET_LABEL[(r.target, r.subset)]} | {r.auprc_a:.3f} | {r.auprc_b:.3f} | "
                     f"{r['diff']:+.3f} ({r.diff_lo:+.3f} to {r.diff_hi:+.3f}) | {r.p_a_better:.2f} |")
    lines += ["", "![EHR value](figures/fig_ehr_gain.png)", "",
              "## Lead time before events", "",
              "An episode starts at the first reading past the threshold after >= 30 min clear of it. *Detected* = at least",
              "one alert in the 2 h before onset at the given false-alert rate (FAR, set on onset windows). Predictions are",
              "every 30 min, so lead time has 30-min resolution and **is capped at 120 min by the horizon**. At 10% FAR",
              "(about 5 alerting windows per patient-day) detection is lenient and lead times saturate near the cap even",
              "for persistence; the 5% rows are more informative. This is a coarse, optimistic measure.", "",
              f"**Simulator limitation.** In the 2 h before a hypo onset the lowest CGM reading is already "
              f"{lead[lead.target == 'hypo'].median_prior_2h_extreme_mgdl.iloc[0]:.0f} mg/dL (median), and before a hyper",
              f"onset the highest is {lead[lead.target == 'hyper'].median_prior_2h_extreme_mgdl.iloc[0]:.0f} mg/dL: simulated "
              "lows arrive as slow drifts, so they are easy to see coming. Real lows (exercise, mistimed insulin) can be",
              "much faster; expect real-world detection and lead time to be markedly worse.", "",
              "| Model | Target | FAR | Episodes | Detected | Median lead (min) | Mean lead (min) |", "|---|---|---|---|---|---|---|"]
    for _, r in lead.sort_values(["target", "false_alert_rate"], kind="stable").iterrows():
        lines.append(f"| {MODELS[r.model][0]} | {r.target} | {r.false_alert_rate:.0%} | {r.n_events} | {r.detection_rate:.0%} | "
                     f"{r.median_lead_min:.0f} | {r.mean_lead_min:.0f} |")
    lines += ["", "## Calibration (all test windows)", "",
              "ECE = count-weighted mean |observed - predicted| over quantile bins (deciles for hyper; upper-tail bins for",
              "the rare hypo target, where deciles collapse to ~0). The fusion model's hypo head is trained",
              "with up-weighted positives (pos_weight = sqrt(neg/pos)), so its raw probabilities overstate hypo risk; they",
              "rank well but should be recalibrated before being shown as absolute probabilities.", "",
              "| Model | Target | Brier | ECE | Mean predicted | Observed rate |", "|---|---|---|---|---|---|"]
    for _, r in cal.iterrows():
        lines.append(f"| {MODELS[r.model][0]} | {r.target} | {r.brier:.4f} | {r.ece:.4f} | {r.mean_predicted:.3f} | {r.prevalence:.3f} |")
    lines += ["", "![Calibration](figures/fig_calibration.png)", "", "![PR curves](figures/fig_pr_curves.png)", "",
              "Full per-subset metrics (incl. *all-window* and sensitivity / PPV / alert burden at 5% and 10% FAR):",
              "[ablation_results.csv](ablation_results.csv).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    test_w, _, scores, ds = load_predictions(cfg)
    docs = resolve_path(cfg["paths"]["docs_dir"])
    figs = resolve_path(Path(cfg["paths"]["docs_dir"]) / "figures")
    print("  discrimination ...", flush=True)
    disc = discrimination(test_w, scores, cfg)
    print("  paired EHR comparisons ...", flush=True)
    ev = ehr_value(test_w, scores, cfg)
    cal, curves = calibration(test_w, scores)
    print("  lead times ...", flush=True)
    lead = lead_times(test_w, scores, ds, cfg)

    disc.to_csv(docs / "ablation_results.csv", index=False)
    ev.to_csv(docs / "ablation_ehr_value.csv", index=False)
    cal.to_csv(docs / "ablation_calibration.csv", index=False)
    lead.to_csv(docs / "ablation_lead_time.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    plot_auprc(disc, figs / "fig_auprc_ablation.png")
    plot_ehr_gain(ev, figs / "fig_ehr_gain.png")
    plot_pr_curves(test_w, scores, figs / "fig_pr_curves.png")
    plot_calibration(curves, figs / "fig_calibration.png")
    write_markdown(disc, ev, cal, lead, docs / "ablation_results.md")
    return {"discrimination": disc, "ehr_value": ev, "calibration": cal, "lead_time": lead}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 ablation report")
    ap.add_argument("--config", default=None)
    out = run(load_config(ap.parse_args().config))
    ev = out["ehr_value"]
    print(ev[["comparison", "target", "subset", "auprc_a", "auprc_b", "diff", "diff_lo", "diff_hi", "p_a_better"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(out["lead_time"].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(out["calibration"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
