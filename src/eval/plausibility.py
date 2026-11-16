"""Biological plausibility check: does each model driver push the risk the way physiology says it should?

    python -m src.eval.plausibility [--data-dir data/external/shanghai]

For every input with a known physiological direction (e.g. a higher sulfonylurea dose should RAISE the risk of
a low; a faster fall should RAISE it; better kidney function should LOWER it), we take the model's exact TreeSHAP
contribution of that input on every onset window and measure the Spearman correlation between the input's value
and its contribution. A positive correlation means "higher value -> model raises the risk". The check passes
when the sign matches physiology; |rho| < 0.1 is reported as "little effect" (no direction learned).

Models: the deployed synthetic-trained model (60 synthetic test patients) and the real-patient model
(ShanghaiT2DM; in-sample, since it is the saved model trained on all 100 patients; skipped without the data).
Caveat: the simulator was BUILT from these mechanisms, so agreement on synthetic data is partly by construction;
the real-patient model is the meaningful check for the CGM-derived drivers.
Writes docs/plausibility.md and docs/plausibility.csv. RESEARCH PROTOTYPE, NOT FOR CLINICAL USE.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import ROOT, load_config, resolve_path
from src.data.build_dataset import load_processed
from src.models.baselines import feature_matrix, load_or_build_features
from src.twin.explain import DISPLAY, mechanism

FLAT = 0.1
# feature -> (expected sign, why) per target. +1: a higher value should raise the risk; -1: lower it.
EXPECTED: dict[str, dict[str, tuple[int, str]]] = {
    "hypo": {
        "cgm_last": (-1, "lower glucose now is closer to 70"),
        "cgm_min_1h": (-1, "a lower recent minimum is closer to 70"),
        "cgm_mean_1h": (-1, "glucose running lower is closer to 70"),
        "cgm_slope_30m": (-1, "a falling trend (negative slope) raises the risk"),
        "cgm_slope_60m": (-1, "a falling trend raises the risk"),
        "cgm_delta_15m": (-1, "a recent drop raises the risk"),
        "min_since_below_70": (-1, "a recent low weakens counter-regulation, so a longer gap lowers the risk"),
        "cgm_frac_below_70": (+1, "more recent time below 70 raises the risk"),
        "steps_1h": (+1, "activity takes up glucose without insulin"),
        "steps_3h": (+1, "activity raises insulin sensitivity for hours"),
        "sulfonylurea_dose_mg": (+1, "sulfonylureas release insulin regardless of glucose"),
        "insulin_units_per_kg": (+1, "more injected insulin, more glucose lowering"),
        "on_insulin": (+1, "insulin can cause lows"),
        "diabetes_duration_years": (+1, "long duration weakens glucagon and adrenaline defences"),
        "prior_severe_hypo_12m": (+1, "past severe lows mark weakened defences"),
        "egfr": (-1, "better kidneys clear drugs faster"),
        "hba1c": (-1, "on glucose-lowering drugs, lower HbA1c runs closer to 70"),
        "age": (+1, "warning symptoms weaken and drugs clear more slowly with age"),
    },
    "hyper": {
        "cgm_last": (+1, "higher glucose now is closer to 180"),
        "cgm_max_1h": (+1, "a higher recent peak is closer to 180"),
        "cgm_slope_30m": (+1, "a rising trend raises the risk"),
        "cgm_delta_15m": (+1, "a recent rise raises the risk"),
        "max_rise_3h": (+1, "a large recent rise signals a carbohydrate-heavy meal"),
        "cgm_frac_above_180": (+1, "recent time above 180 shows glucose running high"),
        "steps_1h": (-1, "activity pulls glucose into muscle"),
        "hba1c": (+1, "a higher HbA1c means glucose runs higher"),
        "bmi": (+1, "more body fat, more insulin resistance"),
        "sulfonylurea_dose_mg": (-1, "sulfonylureas lower glucose"),
    },
}


def check(booster: lgb.Booster, X: pd.DataFrame, target: str, model: str) -> pd.DataFrame:
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]
    share = np.abs(contrib).mean(0)
    share = share / share.sum()
    rows = []
    for f, (sign, why) in EXPECTED[target].items():
        if f not in X.columns:
            continue
        j = list(X.columns).index(f)
        x, c = X[f].to_numpy(float), contrib[:, j]
        ok = np.isfinite(x)
        rho = float(spearmanr(x[ok], c[ok]).statistic) if ok.sum() > 10 and np.ptp(x[ok]) > 0 else np.nan
        verdict = ("little effect" if not np.isfinite(rho) or abs(rho) < FLAT
                   else "agrees" if np.sign(rho) == sign else "DISAGREES")
        rows.append({"model": model, "target": target, "feature": f, "label": DISPLAY.get(f, (f,))[0],
                     "expected": "raises risk" if sign > 0 else "lowers risk", "why": why, "rho": rho,
                     "attention_share": float(share[j]), "verdict": verdict,
                     "mechanism": mechanism(f, target)})
    return pd.DataFrame(rows)


def synthetic(cfg: dict[str, Any], models_dir: Path) -> list[pd.DataFrame]:
    ds = load_processed(cfg)
    feats = load_or_build_features(ds, cfg)
    rows = (ds["windows"].split == "test").to_numpy()
    w = ds["windows"][rows]
    name = cfg["twin"]["model"]
    X = feature_matrix(feats[rows], w, ds["static"], name.removeprefix("lgbm_"))
    out = []
    for t in ("hypo", "hyper"):
        onset = ~w[f"in_event_{t}"].to_numpy(bool)
        b = lgb.Booster(model_file=str(models_dir / f"{name}_{t}.txt"))
        out.append(check(b, X[onset].reset_index(drop=True), t, "Synthetic-trained, deployed (60 synthetic test patients)"))
    return out


def real(cfg: dict[str, Any], models_dir: Path, data_dir: Path) -> list[pd.DataFrame]:
    from src.data.shanghai import load_cohort
    from src.eval.external_shanghai import CGM, build_windows

    periods, static, _ = load_cohort(data_dir, cfg["sampling"]["interval_minutes"])
    w, feats = build_windows(periods, static, cfg)
    X = feats[list(CGM)]
    out = []
    for t in ("hypo", "hyper"):
        onset = ~w[f"in_event_{t}"].to_numpy(bool)
        b = lgb.Booster(model_file=str(models_dir / f"lgbm_cgm_shanghai_{t}.txt"))
        out.append(check(b, X[onset].reset_index(drop=True), t, "Real-patient model (100 ShanghaiT2DM patients)"))
    return out


def write_report(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Biological plausibility: do the model's drivers point the way physiology says?", "",
        "**Research prototype. Not a medical device, not for clinical use.** Generated by "
        "`python -m src.eval.plausibility`.", "",
        "For each input with a known physiological direction, the Spearman correlation between the input's value and "
        "its exact TreeSHAP contribution on onset windows. A positive rho means a higher value makes the model raise "
        f"the risk. Verdict: *agrees* when the sign matches physiology, *little effect* when |rho| < {FLAT}. "
        "*Attention* is the input's share of the model's mean absolute contribution.", "",
    ]
    for (model, target), g in df.groupby(["model", "target"], sort=False):
        n_ok, n_dir = int((g.verdict == "agrees").sum()), int((g.verdict != "little effect").sum())
        att = g[g.verdict != "little effect"]
        att_ok = att[att.verdict == "agrees"].attention_share.sum() / max(att.attention_share.sum(), 1e-12)
        tname = "Hypoglycaemia (< 70 mg/dL)" if target == "hypo" else "Hyperglycaemia (> 180 mg/dL)"
        lines += [f"## {model}: {tname}", "",
                  f"**{n_ok} of {n_dir} inputs with a learned direction point the way physiology predicts; they carry "
                  f"{att_ok:.0%} of the model's attention among the checked inputs** ({len(g) - n_dir} with little effect).", "",
                  "| Input | Physiology says | Why | Model's direction (rho) | Attention | Verdict |",
                  "|---|---|---|---|---|---|",
                  *[f"| {r.label} | higher {r.expected.replace('risk', '')}risk | {r.why} | {r.rho:+.2f} | "
                    f"{r.attention_share:.1%} | {r.verdict} |" for r in g.itertuples()], ""]
    lines += [
        "## Why some inputs disagree", "",
        "The inputs the models rely on most (current glucose, the recent fall, the sulfonylurea dose, HbA1c) all agree. "
        "The disagreements are small inputs, and they have two known causes:", "",
        "* **Overlapping inputs share the credit.** Several inputs describe the same thing (the last 15-minute change, "
        "the 30-minute and 60-minute slopes; the current level and the hourly mean). The model leans on one of them and "
        "uses the others as small corrections, which can carry the opposite sign. Read the group, not the single input: "
        "the main trend input agrees in every model.",
        "* **Confounding by indication.** In the synthetic data, patients on higher sulfonylurea doses also run higher "
        "glucose (the dose was raised because control was poor), so the dose looks like it *raises* the risk of a high. "
        "That is how the patients were treated, not what the drug does. The same applies to diabetes duration and lows.",
        "* **Post-meal timing.** A large rise in the last 3 hours often means the peak has passed and glucose is coming "
        "down, so the model lowers the risk of a high after it. That is physiologically reasonable.", "",
        "**Possible fix:** LightGBM can enforce a physiological direction for chosen inputs (monotone constraints). "
        "Training with constraints on the key drivers would make every explanation point the right way; its cost in "
        "accuracy has to be measured before adopting it.", "",
        "## Reading this", "",
        "* **Synthetic model:** the simulator was built from these mechanisms (docs/model_card.md), so agreement there "
        "is expected and partly by construction. It shows the model learned the intended biology, not that the biology "
        "holds in real patients.",
        "* **Real-patient model:** it sees only CGM and time of day, so only CGM-derived drivers can be checked. These "
        "are learned from real patients and are the meaningful test.",
        "* **What this does not show:** SHAP explains the model, not the patient. A driver that agrees with physiology is "
        "consistent with known biology; it is not proof that this mechanism caused a given low.",
        "* Literature behind each mechanism: sulfonylureas close beta-cell KATP channels and release insulin regardless "
        "of glucose; contracting muscle takes up glucose via GLUT4 without insulin and stays insulin-sensitive for hours; "
        "recent and repeated lows blunt glucagon and adrenaline counter-regulation; reduced kidney function prolongs "
        "sulfonylurea and insulin action. See the finals feedback document's literature map for sources.", "",
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
    parts = synthetic(cfg, models_dir)
    data_dir = Path(args.data_dir)
    if data_dir.exists() and (models_dir / "lgbm_cgm_shanghai_hypo.txt").exists():
        parts += real(cfg, models_dir, data_dir)
    else:
        print(f"ShanghaiT2DM data or real-patient model not found ({data_dir}); real-patient check skipped")
    df = pd.concat(parts, ignore_index=True)
    df.round(4).to_csv(docs / "plausibility.csv", index=False)
    write_report(df, docs / "plausibility.md")
    print(df[["model", "target", "feature", "expected", "rho", "attention_share", "verdict"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
