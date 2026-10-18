"""Does synthetic data help when real data is scarce? Transfer experiment on CGMacros T2D, hyper onset.

    python -m src.eval.transfer_cgmacros --data-dir data/external/cgmacros [--synthetic-dir ...] [--draws 5]

Same cohort, windows, labels, features (CGM + time) and LightGBM hyperparameters as
src/eval/external_cgmacros.py. Models (all scored out-of-fold, leave-one-participant-out):
  * persistence           trend extrapolation, no training
  * zero_shot             LightGBM trained on SYNTHETIC train patients only, applied unchanged
  * real_only             LightGBM trained on real participants only (the existing LOPO baseline)
  * finetune              synthetic model + extra boosting rounds on the real training participants
                          (continued training from the synthetic model's raw scores, i.e. init_model)
  * pooled                one LightGBM on synthetic train windows + real training windows, real windows
                          up-weighted so both sources carry the same total weight
  * pooled_unweighted     the same with every window weighted 1 (full-data comparison only)
Every model that trains on real data early-stops on the same 3 held-out real participants.

Learning curve: for each held-out participant and each random draw, k of the 10 remaining training
participants are used (nested: the k=2 set is inside the k=4 set, ...). Written to docs/: transfer_*.csv
(aggregate metrics only) and docs/figures/fig_transfer_learning_curve.png. See docs/transfer_experiment.md.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ROOT, load_config, resolve_path
from src.data.cgmacros import DEFAULT_SENSOR, build_windows, load_cohort
from src.eval.external_cgmacros import FEATURE_SET, LOPO_INNER_VAL, calibration_cols
from src.eval.metrics import auprc, auroc, evaluate_scores, paired_bootstrap_diff
from src.models.baselines import FEATURE_SETS, lgbm_params, persistence_scores, train_lgbm

TARGET = "hyper"
K_VALUES = (2, 4, 7, 10)          # real training participants in the learning curve (10 = full LOPO)
LC_METHODS = ("real_only", "finetune", "pooled")
FULL_METHODS = ("real_only", "finetune", "pooled", "pooled_unweighted")
PAIRS = (("finetune", "real_only"), ("pooled", "real_only"), ("pooled_unweighted", "real_only"),
         ("finetune", "zero_shot"), ("pooled", "zero_shot"), ("real_only", "zero_shot"))


# --------------------------------------------------------------------------- training helpers
def train_lgbm_transfer(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray, cfg: dict[str, Any],
                        weight: np.ndarray | None = None, init_tr: np.ndarray | None = None,
                        init_va: np.ndarray | None = None) -> lgb.Booster:
    """train_lgbm with optional sample weights and a starting raw score (log-odds) per row.

    With init scores from a pretrained model this is continued training: the new trees boost from the
    pretrained model's predictions (what lgb.train(init_model=...) does). The validation set gets its own
    init scores so early stopping judges the combined model. Predict with `transfer_proba`.
    """
    p = cfg["models"]["lgbm"]
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=weight, init_score=init_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, init_score=init_va, reference=dtr, free_raw_data=False)
    return lgb.train(lgbm_params(cfg), dtr, num_boost_round=p["n_estimators"], valid_sets=[dva],
                     callbacks=[lgb.early_stopping(p["early_stopping_rounds"], verbose=False)])


def transfer_proba(booster: lgb.Booster, X: pd.DataFrame, init: np.ndarray | None = None) -> np.ndarray:
    """Probability from a booster trained by train_lgbm_transfer (add back the init raw score if any)."""
    raw = booster.predict(X, num_iteration=booster.best_iteration, raw_score=True)
    if init is not None:
        raw = raw + init
    return 1.0 / (1.0 + np.exp(-raw))


def balanced_weight(n_synthetic: int, n_real: int) -> float:
    """Per-window weight for real windows so real and synthetic carry equal total weight."""
    return n_synthetic / max(n_real, 1)


def draw_participants(rest: np.ndarray, k_values: tuple[int, ...], n_val: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Early-stopping participants and nested training sets of each size k from `rest`."""
    if max(k_values) + n_val > len(rest):
        raise ValueError(f"need {max(k_values) + n_val} participants, have {len(rest)}")
    perm = rng.permutation(rest)
    return perm[:n_val], {k: perm[n_val:n_val + k] for k in k_values}


def mean_over_draws(fn: Callable[[np.ndarray, np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], float]:
    """fn averaged over the columns of a (windows x draws) score matrix; 1-D scores are one draw."""
    def f(y: np.ndarray, s: np.ndarray) -> float:
        s = s.reshape(len(y), -1)
        return float(np.mean([fn(y, s[:, j]) for j in range(s.shape[1])]))
    return f


# --------------------------------------------------------------------------- experiment
class Transfer:
    """Real T2D windows, synthetic training windows and the synthetic (zero-shot) model."""

    def __init__(self, X: pd.DataFrame, y: np.ndarray, pids: np.ndarray, X_syn: pd.DataFrame, y_syn: np.ndarray,
                 syn_booster: lgb.Booster, cfg: dict[str, Any]):
        self.X, self.y, self.pids, self.cfg = X, y, pids, cfg
        self.X_syn, self.y_syn = X_syn, y_syn
        # the synthetic model's log-odds on every real window: the starting point for fine-tuning
        self.base_raw = syn_booster.predict(X, num_iteration=syn_booster.best_iteration, raw_score=True)

    def fit_score(self, method: str, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> np.ndarray:
        """Train `method` on participants `train` (early stopping on `val`), score participants `test`."""
        tr, va, te = (np.isin(self.pids, g) for g in (train, val, test))
        X, y, cfg = self.X, self.y, self.cfg
        if method == "real_only":
            b = train_lgbm(X[tr], y[tr], X[va], y[va], cfg)
            return b.predict(X[te], num_iteration=b.best_iteration)
        if method == "finetune":
            b = train_lgbm_transfer(X[tr], y[tr], X[va], y[va], cfg, init_tr=self.base_raw[tr], init_va=self.base_raw[va])
            return transfer_proba(b, X[te], self.base_raw[te])
        if method in ("pooled", "pooled_unweighted"):
            w_real = balanced_weight(len(self.y_syn), tr.sum()) if method == "pooled" else 1.0
            Xp = pd.concat([self.X_syn, X[tr]], ignore_index=True)
            yp = np.concatenate([self.y_syn, y[tr]])
            wp = np.concatenate([np.ones(len(self.y_syn)), np.full(tr.sum(), w_real)])
            b = train_lgbm_transfer(Xp, yp, X[va], y[va], cfg, weight=wp)
            return transfer_proba(b, X[te])
        raise ValueError(method)

    def full_lopo(self, methods: tuple[str, ...] = FULL_METHODS) -> dict[str, np.ndarray]:
        """Out-of-fold scores with exactly the participant split of external_cgmacros.lopo_scores."""
        uniq = np.unique(self.pids)
        out = {m: np.full(len(self.y), np.nan) for m in methods}
        for i, p in enumerate(uniq):
            rest = uniq[uniq != p]
            val = np.random.default_rng([self.cfg["seed"], i]).choice(rest, LOPO_INNER_VAL, replace=False)
            train = rest[~np.isin(rest, val)]
            for m in methods:
                out[m][self.pids == p] = self.fit_score(m, train, val, np.array([p]))
            print(f"  full LOPO fold {i + 1}/{len(uniq)}", flush=True)
        return out

    def learning_curve(self, k_values: tuple[int, ...], n_draws: int,
                       methods: tuple[str, ...] = LC_METHODS) -> dict[tuple[str, int], np.ndarray]:
        """(method, k) -> out-of-fold scores, one column per random draw of training participants."""
        uniq = np.unique(self.pids)
        out = {(m, k): np.full((len(self.y), n_draws), np.nan) for m in methods for k in k_values}
        t0 = time.time()
        for r in range(n_draws):
            for i, p in enumerate(uniq):
                rng = np.random.default_rng([self.cfg["seed"], 1 + r, i])
                val, trains = draw_participants(uniq[uniq != p], k_values, LOPO_INNER_VAL, rng)
                te = self.pids == p
                for k, train in trains.items():
                    for m in methods:
                        out[(m, k)][te, r] = self.fit_score(m, train, val, np.array([p]))
            print(f"  learning curve draw {r + 1}/{n_draws} ({time.time() - t0:.0f} s)", flush=True)
        return out


# --------------------------------------------------------------------------- metrics
def metric_rows(windows: pd.DataFrame, scores: dict[str, np.ndarray], cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for subset in ("onset", "all"):
        m = np.ones(len(windows), bool) if subset == "all" else ~windows[f"in_event_{TARGET}"].to_numpy(bool)
        for name, s in scores.items():
            r = evaluate_scores(windows[TARGET].to_numpy()[m], s[m], windows.patient_id.to_numpy()[m], cfg)
            extra = calibration_cols(windows, {TARGET: s}, TARGET, subset) if name != "persistence" else {}
            rows.append({"target": TARGET, "subset": subset, "model": name, **r, **extra})
    return pd.DataFrame(rows)


def paired_rows(y: np.ndarray, g: np.ndarray, scores: dict[str, np.ndarray], pairs: tuple[tuple[str, str], ...],
                cfg: dict[str, Any], **keys: Any) -> list[dict[str, Any]]:
    """Paired patient-bootstrap A - B; 2-D scores are averaged over draws inside every resample."""
    rows = []
    for a, b in pairs:
        if a not in scores or b not in scores:
            continue
        for metric, fn in (("auprc", auprc), ("auroc", auroc)):
            d = paired_bootstrap_diff(y, scores[a], scores[b], g, mean_over_draws(fn), cfg["eval"]["n_bootstrap"], cfg["seed"])
            rows.append({**keys, "model_a": a, "model_b": b, "metric": metric, **d})
    return rows


def learning_curve_tables(windows: pd.DataFrame, lc: dict[tuple[str, int], np.ndarray], refs: dict[str, np.ndarray],
                          cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-draw onset metrics and paired (augmented - real_only, mean over draws) differences per k."""
    m = ~windows[f"in_event_{TARGET}"].to_numpy(bool)
    y, g = windows[TARGET].to_numpy(int)[m], windows.patient_id.to_numpy()[m]
    rows, paired = [], []
    for (method, k), S in lc.items():
        for r in range(S.shape[1]):
            rows.append({"subset": "onset", "model": method, "k": k, "draw": r,
                         "auprc": auprc(y, S[m, r]), "auroc": auroc(y, S[m, r])})
    for k in sorted({k for _, k in lc}):
        scores = {meth: S[m] for (meth, kk), S in lc.items() if kk == k} | {n: s[m] for n, s in refs.items()}
        paired += paired_rows(y, g, scores, (("finetune", "real_only"), ("pooled", "real_only"),
                                             ("real_only", "zero_shot"), ("finetune", "zero_shot"),
                                             ("pooled", "zero_shot")), cfg, subset="onset", k=k)
    return pd.DataFrame(rows), pd.DataFrame(paired)


# --------------------------------------------------------------------------- figure
LC_STYLE = {"real_only": ("Real only", "#2a78d6", "o"), "finetune": ("Synthetic pretrain + real fine-tune", "#eb6834", "s"),
            "pooled": ("Pooled synthetic + real (real up-weighted)", "#1baf7a", "D")}


def plot_learning_curve(lc_rows: pd.DataFrame, lc_paired: pd.DataFrame, refs: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.eval.ablation import GRID, INK, INK_2, INK_3, SURFACE, _style

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    agg = lc_rows.groupby(["model", "k"]).auprc.agg(["mean", "min", "max"]).reset_index()
    offsets = {"real_only": -0.15, "finetune": 0.0, "pooled": 0.15}
    for meth, (label, color, marker) in LC_STYLE.items():
        d = agg[agg.model == meth].sort_values("k")
        x = d.k + offsets[meth]
        ax.vlines(x, d["min"], d["max"], color=color, lw=2, alpha=0.5)
        ax.plot(x, d["mean"], color=color, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5, label=label)
    ks = sorted(agg.k.unique())
    for name, label, ls in (("zero_shot", "Synthetic only (zero-shot)", "--"), ("persistence", "Persistence", ":")):
        v = refs.loc[refs.model == name, "auprc"].iloc[0]
        ax.axhline(v, color=INK_3, ls=ls, lw=1.2)
        ax.text(max(ks) + 0.55, v + 0.002, f"{label} {v:.3f}", fontsize=7, color=INK_2, va="bottom", ha="right")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(1.5, max(ks) + 0.6)
    ax.set_xlabel("Real T2D participants used for training (+3 for early stopping)", fontsize=8, color=INK_2)
    ax.set_ylabel("AUPRC, hyper onset (mean over draws; bar = min-max)", fontsize=8, color=INK_2)
    ax.set_title("Performance vs amount of real data", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax = axes[1]
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axhline(0, color=INK_3, lw=1)
    d0 = lc_paired[(lc_paired.metric == "auprc") & (lc_paired.model_b == "real_only")]
    for meth, dx in (("finetune", -0.08), ("pooled", 0.08)):
        label, color, marker = LC_STYLE[meth]
        d = d0[d0.model_a == meth].sort_values("k")
        x = d.k + dx
        ax.vlines(x, d.diff_lo, d.diff_hi, color=color, lw=2)
        ax.scatter(x, d["diff"], s=50, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=label.split(" (")[0])
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(1.5, max(ks) + 0.6)
    ax.set_xlabel("Real T2D participants used for training", fontsize=8, color=INK_2)
    ax.set_ylabel("AUPRC difference vs real only (paired 95% CI)", fontsize=8, color=INK_2)
    ax.set_title("Gain from adding synthetic data (above zero = helps)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("CGMacros T2D (14 real participants, leave-one-participant-out). Not a clinical validation.",
                 fontsize=9, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- runner
def load_synthetic(syn_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic windows + cached handcrafted features (window_features.npz, written by src.models.baselines)."""
    z = np.load(syn_dir / "window_features.npz", allow_pickle=False)
    feats = pd.DataFrame(z["values"], columns=z["columns"])
    windows = pd.read_csv(syn_dir / "windows.csv.gz")
    if len(feats) != len(windows):
        raise ValueError(f"{syn_dir}: window_features.npz and windows.csv.gz have different row counts")
    return windows, feats


def run(cfg: dict[str, Any], data_dir: Path, syn_dir: Path, sensor: str, n_draws: int,
        k_values: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    cols = list(FEATURE_SETS[FEATURE_SET])
    cohort = load_cohort(data_dir, sensor=sensor, statuses=("t2d",), dt_min=cfg["sampling"]["interval_minutes"])
    windows, feats = build_windows(cohort, cfg)
    X, y = feats[cols].reset_index(drop=True), windows[TARGET].to_numpy(int)

    sw, sf = load_synthetic(syn_dir)
    tr, va = (sw.split == "train").to_numpy(), (sw.split == "val").to_numpy()
    Xs, ys = sf[cols], sw[TARGET].to_numpy(int)
    syn_booster = train_lgbm(Xs[tr], ys[tr], Xs[va], ys[va], cfg)   # = lgbm_cgm_hyper.txt (same data, seed)
    print(f"  synthetic model: best_iter={syn_booster.best_iteration}, {tr.sum()} synthetic train windows", flush=True)

    exp = Transfer(X, y, windows.patient_id.to_numpy(), Xs[tr].reset_index(drop=True), ys[tr], syn_booster, cfg)
    horizon_min = cfg["windows"]["horizon_hours"] * 60
    refs = {"persistence": persistence_scores(feats, horizon_min)[TARGET],
            "zero_shot": syn_booster.predict(X, num_iteration=syn_booster.best_iteration)}
    full = refs | exp.full_lopo()
    res = metric_rows(windows, full, cfg)
    paired = []
    for subset in ("onset", "all"):
        m = np.ones(len(windows), bool) if subset == "all" else ~windows[f"in_event_{TARGET}"].to_numpy(bool)
        paired += paired_rows(y[m], windows.patient_id.to_numpy()[m], {n: s[m] for n, s in full.items()}, PAIRS,
                              cfg, target=TARGET, subset=subset)

    lc = exp.learning_curve(k_values, n_draws)
    lc_rows, lc_paired = learning_curve_tables(windows, lc, refs, cfg)
    return {"metrics": res, "paired": pd.DataFrame(paired), "lc": lc_rows, "lc_paired": lc_paired}


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic-to-real transfer experiment on CGMacros T2D")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "external" / "cgmacros"))
    ap.add_argument("--synthetic-dir", default=None, help="folder with window_features.npz + windows.csv.gz; "
                                                          "defaults to paths.processed_dir")
    ap.add_argument("--sensor", default=DEFAULT_SENSOR, choices=("dexcom", "libre"))
    ap.add_argument("--draws", type=int, default=5, help="random participant draws per learning-curve point")
    ap.add_argument("--k", default=",".join(map(str, K_VALUES)), help="comma-separated training-set sizes")
    ap.add_argument("--out-dir", default=None, help="defaults to paths.docs_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    syn_dir = Path(args.synthetic_dir) if args.synthetic_dir else resolve_path(cfg["paths"]["processed_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else resolve_path(cfg["paths"]["docs_dir"])
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    k_values = tuple(int(k) for k in args.k.split(","))
    out = run(cfg, Path(args.data_dir), syn_dir, args.sensor, args.draws, k_values)

    out["metrics"].to_csv(out_dir / "transfer_experiment.csv", index=False)
    out["paired"].to_csv(out_dir / "transfer_experiment_paired.csv", index=False)
    out["lc"].to_csv(out_dir / "transfer_learning_curve.csv", index=False)
    out["lc_paired"].to_csv(out_dir / "transfer_learning_curve_paired.csv", index=False)
    refs = out["metrics"][out["metrics"].subset == "onset"]
    plot_learning_curve(out["lc"], out["lc_paired"], refs, out_dir / "figures" / "fig_transfer_learning_curve.png")

    pd.set_option("display.width", 250)
    cols = ["subset", "model", "n_patients", "n_positive", "prevalence", "auprc", "auprc_lo", "auprc_hi", "auroc",
            "auroc_lo", "auroc_hi", "sens_at_far10", "ppv_at_far10", "mean_predicted", "ece"]
    print(out["metrics"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(out["paired"].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    agg = out["lc"].groupby(["model", "k"])[["auprc", "auroc"]].agg(["mean", "min", "max"])
    print(agg.to_string(float_format=lambda v: f"{v:.3f}"))
    print(out["lc_paired"].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
