"""Subgroup performance of the deployed model (fairness / robustness check).

    python -m src.eval.subgroups        # reads saved test predictions; writes docs/subgroups.{md,csv}

For each subgroup (sex, age band, kidney function, drug class, HbA1c band, BMI band) and target, on test
patients' ONSET windows: AUROC and AUPRC with patient-bootstrap 95% CIs, AUPRC lift over the group's
prevalence (AUPRC is not comparable across groups with different prevalence; lift is a fairer read),
and calibration (mean predicted vs observed). Groups with too few events are suppressed rather than
reported as noise. SYNTHETIC DATA: subgroup structure comes from the simulator, not real populations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.eval.metrics import auprc, auroc, evaluation_subsets, patient_bootstrap

MIN_POSITIVES, MIN_POSITIVE_PATIENTS = 20, 3


def subgroup_labels(ehr: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: the subgroup each patient belongs to, per dimension."""
    drug = np.where(ehr.on_insulin, "Insulin", np.where(ehr.on_sulfonylurea, "Sulfonylurea (no insulin)", "Neither"))
    return pd.DataFrame({
        "Sex": np.where(ehr.sex == "M", "Male", "Female"),
        "Age": pd.cut(ehr.age, [0, 49, 64, 200], labels=["< 50", "50-64", ">= 65"]).astype(str).to_numpy(),
        "Kidney function (eGFR)": np.where(ehr.egfr < 60, "< 60 (reduced)", ">= 60"),
        "Drug class": drug,
        "HbA1c": pd.cut(ehr.hba1c, [0, 6.95, 8.95, 100], labels=["< 7%", "7-8.9%", ">= 9%"]).astype(str).to_numpy(),
        "BMI": np.where(ehr.bmi < 25, "< 25", ">= 25"),
    }, index=ehr.patient_id.to_numpy() if "patient_id" in ehr else ehr.index)


def subgroup_table(windows: pd.DataFrame, scores: dict[str, np.ndarray], groups: pd.DataFrame,
                   cfg: dict[str, Any]) -> pd.DataFrame:
    """Metrics per (target, dimension, group) on onset windows. `windows` must have patient_id."""
    masks = {(t, s): m for t, s, m in evaluation_subsets(windows)}
    rows = []
    for target in ("hyper", "hypo"):
        onset = masks[(target, "onset")]
        w = windows[onset]
        y_all, s_all = w[target].to_numpy(int), scores[target][onset]
        pid = w.patient_id.to_numpy()
        for dim in groups.columns:
            g_of = groups[dim].reindex(pid).to_numpy()
            for g in sorted(pd.unique(g_of)):
                m = g_of == g
                y, s, p = y_all[m], s_all[m], pid[m]
                n_pos, pos_pat = int(y.sum()), int(pd.Series(y).groupby(p).max().sum())
                row = {"target": target, "dimension": dim, "group": g, "patients": int(len(np.unique(p))),
                       "windows": int(m.sum()), "positives": n_pos, "patients_with_events": pos_pat,
                       "prevalence": float(y.mean()) if len(y) else np.nan,
                       "mean_predicted": float(s.mean()) if len(s) else np.nan}
                enough = n_pos >= MIN_POSITIVES and pos_pat >= MIN_POSITIVE_PATIENTS
                row["reported"] = enough
                if enough:
                    for name, fn in (("auroc", auroc), ("auprc", auprc)):
                        row[name] = fn(y, s)
                        row[f"{name}_lo"], row[f"{name}_hi"] = patient_bootstrap(y, s, p, fn, cfg["eval"]["n_bootstrap"], cfg["seed"])
                    row["auprc_lift"] = row["auprc"] / row["prevalence"]
                rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, overall: dict[str, dict[str, float]], path: Path) -> None:
    def ci(r: pd.Series, m: str) -> str:
        return f"{r[m]:.3f} ({r[m + '_lo']:.3f}-{r[m + '_hi']:.3f})"

    lines = [
        "# Subgroup performance (deployed model)", "",
        "**SYNTHETIC DATA.** LightGBM on CGM + wearables + EHR, 60 test patients, **onset** windows. Subgroup",
        "differences reflect how the simulator generates each group, not real-world disparities, but the check",
        "shows how the model would be audited. Groups with fewer than "
        f"{MIN_POSITIVES} positive windows or {MIN_POSITIVE_PATIENTS} patients with events are",
        "not scored (too few to say anything). 95% CIs resample whole patients. *Lift* = AUPRC / prevalence:",
        "AUPRC depends on prevalence, so lift is the fairer comparison across groups. *Calibration* compares",
        "the mean predicted risk with the observed event rate.", "",
    ]
    for target, title in (("hyper", "Hyperglycaemia (> 180 mg/dL in 2 h)"), ("hypo", "Hypoglycaemia (< 70 mg/dL in 2 h)")):
        o = overall[target]
        lines += [f"## {title}", "",
                  f"All test patients: AUROC {o['auroc']:.3f}, AUPRC {o['auprc']:.3f}, prevalence {o['prevalence']:.3f}, "
                  f"lift {o['auprc'] / o['prevalence']:.1f}x.", "",
                  "| Dimension | Group | Patients | Positives (patients) | Prevalence | AUROC (95% CI) | AUPRC (95% CI) | Lift | Mean predicted |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in df[df.target == target].iterrows():
            if r.reported:
                lines.append(f"| {r.dimension} | {r.group} | {r.patients} | {r.positives} ({r.patients_with_events}) | "
                             f"{r.prevalence:.3f} | {ci(r, 'auroc')} | {ci(r, 'auprc')} | {r.auprc_lift:.1f}x | {r.mean_predicted:.3f} |")
            else:
                lines.append(f"| {r.dimension} | {r.group} | {r.patients} | {r.positives} ({r.patients_with_events}) | "
                             f"{r.prevalence:.3f} | *too few events* | | | {r.mean_predicted:.3f} |")
        lines.append("")
    rep = df[df.reported]
    lines += ["## Reading this", ""]
    for target in ("hyper", "hypo"):
        r = rep[rep.target == target]
        if len(r):
            lo, hi = r.loc[r.auroc.idxmin()], r.loc[r.auroc.idxmax()]
            lines.append(f"* **{'Hyper' if target == 'hyper' else 'Hypo'}:** AUROC ranges from {lo.auroc:.3f} "
                         f"({lo.dimension}: {lo.group}) to {hi.auroc:.3f} ({hi.dimension}: {hi.group}) across scored groups.")
    for target in ("hyper", "hypo"):
        r = rep[rep.target == target]
        if len(r) > 1:
            worst = r.loc[r.auprc.idxmin()]
            lines.append(f"* **Lowest {'hyper' if target == 'hyper' else 'hypo'} AUPRC:** {worst.dimension}: {worst.group} "
                         f"({worst.auprc:.3f} at prevalence {worst.prevalence:.3f}, lift {worst.auprc_lift:.1f}x; "
                         f"{worst.patients} patients, {worst.positives} positive windows from {worst.patients_with_events} "
                         "patients). Low AUPRC partly reflects rare events in that group, so read it with AUROC and lift; "
                         "small groups make all of these unstable.")
    tiny = df[(df.target == "hyper") & (df.patients < 5)]
    if len(tiny):
        lines.append("* **Under-represented in the test set:** " + ", ".join(
            f"{r.dimension} {r.group} ({r.patients} patient{'s' if r.patients != 1 else ''})" for _, r in tiny.iterrows())
            + ". These groups cannot be audited at all here, even though the simulator gives them higher hypo risk "
            "by design (e.g. slower drug clearance with low eGFR).")
    hypo_unscored = df[(df.target == "hypo") & ~df.reported]
    if len(hypo_unscored):
        lines.append("* **Hypoglycaemia cannot be audited in every group:** " + ", ".join(
            f"{r.dimension} {r.group}" for _, r in hypo_unscored.iterrows()) + " have too few events. In this simulator, "
            "patients on neither insulin nor a sulfonylurea essentially never go low, which is clinically expected.")
    lines += ["* A real audit would need a much larger, representative cohort, and should add region, diet, "
              "socioeconomic status and device type, which this synthetic EHR does not contain.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict[str, Any]) -> pd.DataFrame:
    from src.data.build_dataset import load_processed
    ds = load_processed(cfg)
    res_dir = Path(resolve_path(cfg["paths"]["processed_dir"])) / "results"
    preds = pd.read_csv(res_dir / "baselines_test_predictions.csv.gz")
    w = ds["windows"].iloc[preds.window_id.to_numpy()].reset_index(drop=True)
    model = cfg["twin"]["model"]
    scores = {t: preds[f"{model}_{t}"].to_numpy() for t in ("hyper", "hypo")}
    ehr = pd.read_csv(resolve_path(cfg["paths"]["raw_dir"]) / "ehr.csv", keep_default_na=False)
    groups = subgroup_labels(ehr)
    df = subgroup_table(w, scores, groups, cfg)
    masks = {(t, s): m for t, s, m in evaluation_subsets(w)}
    overall = {}
    for t in ("hyper", "hypo"):
        m = masks[(t, "onset")]
        y = w[t].to_numpy(int)[m]
        overall[t] = {"auroc": auroc(y, scores[t][m]), "auprc": auprc(y, scores[t][m]), "prevalence": float(y.mean())}
    docs = resolve_path(cfg["paths"]["docs_dir"])
    df.to_csv(docs / "subgroups.csv", index=False)
    write_markdown(df, overall, docs / "subgroups.md")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Subgroup performance of the deployed model")
    ap.add_argument("--config", default=None)
    df = run(load_config(ap.parse_args().config))
    cols = ["target", "dimension", "group", "patients", "positives", "prevalence", "auroc", "auprc", "auprc_lift", "reported"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
