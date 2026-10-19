"""Pack the files the dashboard needs into demo_bundle/ so it can be deployed without running the pipeline.

    python scripts/make_demo_bundle.py        # after scripts/run_pipeline.{sh,cmd}

The dashboard loads demo_bundle/ automatically when data/processed is missing (e.g. on Streamlit
Community Cloud), or always when TWIN_USE_DEMO_BUNDLE=1. The personalised what-if twin needs the patient
log (meals + doses) and the cached parameter fits; the oracle mode regenerates the hidden simulation from
config.yaml + seed. Everything in the bundle is SYNTHETIC.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.twin.state import TwinContext  # noqa: E402

BUNDLE = ROOT / "demo_bundle"
PROCESSED = ("series.npz", "windows.csv.gz", "static_features.csv", "static_encoder.json", "splits.csv",
             "window_features.npz", "dataset_summary.json", "linkage_report.json", "patient_log.csv.gz",
             "twin_fits.csv")


def main() -> None:
    cfg = load_config()
    TwinContext.load(cfg)               # makes sure the feature cache and alert thresholds exist
    src_proc = ROOT / cfg["paths"]["processed_dir"]
    src_models = ROOT / cfg["paths"]["models_dir"]
    model = cfg["twin"]["model"]
    files = [(ROOT / cfg["paths"]["raw_dir"] / "ehr.csv", BUNDLE / "raw" / "ehr.csv")]
    files += [(src_proc / f, BUNDLE / "processed" / f) for f in PROCESSED]
    files += [(src_models / f, BUNDLE / "processed" / "models" / f)
              for f in (f"{model}_hyper.txt", f"{model}_hypo.txt", f"{model}_alert_thresholds.json",
                        "twin_intervals.json")]

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    rows = []
    for src, dst in files:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append((dst.relative_to(BUNDLE).as_posix(), dst.stat().st_size, hashlib.sha256(dst.read_bytes()).hexdigest()[:16]))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    total = sum(r[1] for r in rows)
    lines = ["# Demo bundle", "",
             "**SYNTHETIC DATA. Not for clinical use.** Minimal files for running the dashboard without the",
             "pipeline (e.g. on Streamlit Community Cloud). Rebuild with `python scripts/make_demo_bundle.py`",
             "after re-running the pipeline.", "",
             f"- Built from commit `{commit}`, seed {cfg['seed']}, model `{model}`",
             f"- Total size {total / 1e6:.1f} MB", "",
             "| File | Bytes | SHA-256 (first 16) |", "|---|---|---|",
             *[f"| {name} | {size:,} | `{digest}` |" for name, size, digest in rows], ""]
    (BUNDLE / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} files, {total / 1e6:.1f} MB -> {BUNDLE}")


if __name__ == "__main__":
    main()
