"""Draw docs/architecture.png (same structure as docs/architecture.mmd) with matplotlib.

    python scripts/make_architecture_diagram.py

Edges are routed orthogonally through the gaps between columns ("buses") so no line crosses a box.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyBboxPatch          # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
INK, INK_2, INK_3, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
STYLE = {"ehr": ("#fdebe3", "#eb6834"), "sensor": ("#e3eefb", "#2a78d6"), "neutral": ("#f3f2ee", "#a3a29c")}

W, H = 0.165, 0.135
COLS = [0.025, 0.225, 0.425, 0.625, 0.815]
LANES = ["1  Synthetic data", "2  Linking & features", "3  Models", "4  Evaluation", "5  Digital twin & dashboard"]
ROWS = [0.72, 0.52, 0.32, 0.12]

BOXES = {
    "ehr": (0, ROWS[0], "ehr", "Synthetic EHR cohort", "300 patients: age, BMI, HbA1c,\neGFR, medications, comorbidities"),
    "sim": (0, ROWS[1], "neutral", "Mechanistic simulator", "Indian meals, insulin / SU PK,\ntitration, sleep, activity"),
    "wear": (0, ROWS[2], "sensor", "Wearable stream", "CGM 5-min (lag, noise, dropouts),\nsteps, heart rate, sleep"),
    "enr": (0, ROWS[3], "neutral", "Enrollment log", "patient_id <-> device_id\n(synthetic linkage)"),
    "link": (1, ROWS[0], "neutral", "Linker", "1:1 enrollment join,\nvalidated, orphans reported"),
    "win": (1, ROWS[1], "sensor", "Rolling windows", "6 h history -> 2 h horizon\nhyper > 180, hypo < 70"),
    "stat": (1, ROWS[2], "ehr", "Static vector", "encoded EHR, scaled on\ntrain patients only"),
    "split": (1, ROWS[3], "neutral", "Patient-level split", "65 / 15 / 20 by drug class;\nleakage guard"),
    "pers": (2, ROWS[0], "neutral", "Persistence baseline", "extrapolate last value\nalong 30-min trend"),
    "lgb": (2, 0.42, "neutral", "LightGBM per target", "handcrafted features:\nCGM / + wearables / + EHR"),
    "fus": (2, ROWS[3], "neutral", "Fusion net (PyTorch)", "static MLP + GRU (or 1D-CNN)\n-> shared -> 2 heads"),
    "met": (3, 0.30, "neutral", "Metrics", "AUPRC (hypo headline), AUROC,\nsens @ FAR, lead time,\ncalibration, patient CIs"),
    "abl": (3, 0.08, "neutral", "Ablation", "paired EHR-value test;\ntables + figures -> docs/"),
    "state": (4, ROWS[0], "neutral", "PatientTwin", "profile, 6 h history, risks;\nalerts set on validation"),
    "exp": (4, ROWS[1], "neutral", "Explanations", "TreeSHAP drivers\nper prediction"),
    "what": (4, ROWS[2], "neutral", "What-if engine", "replays the simulator with edits\n(synthetic only) + ML re-predict"),
    "app": (4, ROWS[3], "neutral", "Streamlit dashboard", "timeline, drivers,\nwhat-if, overview"),
}


def box(key: str) -> tuple[float, float, float, float]:
    col, y, *_ = BOXES[key]
    return COLS[col], y, COLS[col] + W, y + H


def cy(key: str) -> float:
    _, y0, _, y1 = box(key)
    return (y0 + y1) / 2


def cx(key: str) -> float:
    x0, _, x1, _ = box(key)
    return (x0 + x1) / 2


def path(ax, pts: list[tuple[float, float]], arrow: bool = True, dashed: bool = False) -> None:
    """Polyline with an arrowhead on its last segment."""
    xs, ys = zip(*pts)
    ls = (0, (4, 3)) if dashed else "-"
    ax.plot(xs[:-1] if arrow else xs, ys[:-1] if arrow else ys, color=INK_3, lw=1.3, ls=ls, zorder=1,
            solid_capstyle="round")
    if arrow:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=INK_3, lw=1.3, mutation_scale=12, shrinkA=0, shrinkB=1,
                                    linestyle=ls))


def down(ax, a: str, b: str) -> None:
    path(ax, [(cx(a), box(a)[1]), (cx(b), box(b)[3])])


def label(ax, x: float, y: float, text: str, ha: str = "left") -> None:
    ax.text(x, y, text, fontsize=8.5, color=INK_2, ha=ha, va="center", zorder=3,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))


def main() -> None:
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.025, 0.955, "Glycaemic digital twin for Type 2 diabetes: architecture", fontsize=17, color=INK,
            fontweight="bold", va="center")
    ax.text(0.025, 0.925, "Static EHR + dynamic wearable streams -> 2-hour risk of hyperglycaemia (> 180 mg/dL) and "
            "hypoglycaemia (< 70 mg/dL)", fontsize=11, color=INK_2, va="center")
    for x, lane in zip(COLS, LANES):
        ax.text(x, 0.885, lane, fontsize=11, color=INK_2, fontweight="bold", va="center")

    # column 1 internal
    down(ax, "ehr", "sim")
    label(ax, cx("ehr") + 0.008, (box("ehr")[1] + box("sim")[3]) / 2, "traits drive dynamics\n(by design)")
    down(ax, "sim", "wear")
    # column 1 -> linker: bus in the gap
    bus1 = COLS[0] + W + 0.017
    for k in ("ehr", "wear", "enr"):
        path(ax, [(COLS[0] + W, cy(k)), (bus1, cy(k))], arrow=False)
    path(ax, [(bus1, cy("enr")), (bus1, cy("ehr"))], arrow=False)
    path(ax, [(bus1, cy("link")), (COLS[1], cy("link"))])
    # column 2 chain
    for a, b in (("link", "win"), ("win", "stat"), ("stat", "split")):
        down(ax, a, b)
    # split -> models bus
    bus2 = COLS[1] + W + 0.017
    path(ax, [(COLS[1] + W, cy("split")), (bus2, cy("split")), (bus2, cy("pers"))], arrow=False)
    for k in ("pers", "lgb", "fus"):
        path(ax, [(bus2, cy(k)), (COLS[2], cy(k))])
    # models -> metrics bus
    bus3 = COLS[2] + W + 0.012
    for k in ("pers", "lgb", "fus"):
        path(ax, [(COLS[2] + W, cy(k)), (bus3, cy(k))], arrow=False)
    path(ax, [(bus3, cy("pers")), (bus3, cy("fus"))], arrow=False)
    path(ax, [(bus3, cy("met")), (COLS[3], cy("met"))])
    down(ax, "met", "abl")
    # deployed model -> twin, routed above the metrics column
    y_dep = box("lgb")[3] + 0.04
    x_dep = COLS[3] + W + 0.012
    path(ax, [(cx("lgb"), box("lgb")[3]), (cx("lgb"), y_dep), (x_dep, y_dep), (x_dep, cy("state")), (COLS[4], cy("state"))])
    label(ax, (COLS[3] + COLS[3] + W) / 2, y_dep + 0.022, "deployed model: LightGBM CGM + wearables + EHR", ha="center")
    # column 5 chain
    for a, b in (("state", "exp"), ("exp", "what"), ("what", "app")):
        down(ax, a, b)

    for key, (col, y, style, title, body) in BOXES.items():
        fill, edge = STYLE[style]
        x = COLS[col]
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.003,rounding_size=0.01",
                                    facecolor=fill, edgecolor=edge, lw=1.6, zorder=2))
        ax.text(x + 0.01, y + H - 0.028, title, fontsize=11, color=INK, fontweight="bold", va="center", zorder=3)
        ax.text(x + 0.01, y + H - 0.056, body, fontsize=9, color=INK_2, va="top", zorder=3, linespacing=1.35)

    for i, (style, text) in enumerate((("ehr", "static / EHR stream"), ("sensor", "dynamic / sensor stream"),
                                       ("neutral", "processing"))):
        fill, edge = STYLE[style]
        x = 0.025 + i * 0.16
        ax.add_patch(FancyBboxPatch((x, 0.025), 0.018, 0.022, boxstyle="round,pad=0.002", facecolor=fill, edgecolor=edge, lw=1.4))
        ax.text(x + 0.026, 0.036, text, fontsize=9, color=INK_2, va="center")
    ax.text(0.975, 0.036, "ALL DATA SYNTHETIC. Research prototype, not for clinical use.", fontsize=9.5, color=INK_2,
            ha="right", va="center", fontweight="bold")
    fig.savefig(OUT, dpi=130, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
