"""Stage 2 orchestration: link streams, split by patient, encode static features, label windows.

    python -m src.data.build_dataset          # reads data/raw, writes data/processed

Outputs (data/processed/):
    series.npz             raw (P, T, C) wearable series + patient_ids + timestamps
    windows.csv.gz         one row per prediction time: labels, nocturnal flag, split
    static_features.csv    scaled static vector per patient (encoder fit on train only)
    static_encoder.json    scaling parameters
    splits.csv             patient -> stratum -> split
    linkage_report.json    result of the EHR <-> device linkage
    dataset_summary.json   window counts and positive rates per split
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.ehr_features import StaticEncoder, encode_static
from src.data.linker import link_by_enrollment
from src.data.wearable_features import SERIES_CHANNELS, label_windows, to_series

SPLITS = ("train", "val", "test")


class PatientLeakageError(AssertionError):
    """A patient appears in more than one split."""


def drug_stratum(ehr: pd.DataFrame) -> pd.Series:
    """Stratify by the main hypoglycaemia driver so every split sees insulin and SU patients."""
    s = np.where(ehr.on_insulin, "insulin", np.where(ehr.on_sulfonylurea, "sulfonylurea", "other"))
    return pd.Series(s, index=ehr.patient_id.to_numpy(), name="stratum")


def patient_split(ehr: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Assign each PATIENT (never a row) to train / val / test, stratified by drug class.

    Deterministic given cfg['seed']. Within each stratum, patients are shuffled and
    the first round(n * test_fraction) go to test, the next round(n * val_fraction)
    to val, and the rest to train.
    """
    rng = np.random.default_rng([cfg["seed"], 100])
    f_test, f_val = cfg["split"]["test_fraction"], cfg["split"]["val_fraction"]
    strata = drug_stratum(ehr)
    rows = []
    for stratum in sorted(strata.unique()):
        ids = np.sort(strata.index[strata == stratum].to_numpy())
        ids = ids[rng.permutation(len(ids))]
        n_test, n_val = int(round(len(ids) * f_test)), int(round(len(ids) * f_val))
        labels = ["test"] * n_test + ["val"] * n_val + ["train"] * (len(ids) - n_test - n_val)
        rows += [(pid, stratum, lab) for pid, lab in zip(ids, labels)]
    return pd.DataFrame(rows, columns=["patient_id", "stratum", "split"]).sort_values("patient_id").reset_index(drop=True)


def assert_no_patient_leakage(windows: pd.DataFrame) -> None:
    """Fail loudly if any patient's windows land in more than one split."""
    n_splits = windows.groupby("patient_id")["split"].nunique()
    leaky = n_splits[n_splits > 1]
    if len(leaky):
        raise PatientLeakageError(f"{len(leaky)} patient(s) appear in multiple splits, e.g. {list(leaky.index[:5])}")


def build_dataset(cfg: dict[str, Any], tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Pure in-memory build from raw tables (ehr, wearables, enrollment)."""
    ehr = tables["ehr"].sort_values("patient_id").reset_index(drop=True)
    linked, report = link_by_enrollment(ehr, tables["wearables"], tables["enrollment"])
    ehr = ehr[ehr.patient_id.isin(linked.patient_id.unique())].reset_index(drop=True)
    patient_ids = ehr.patient_id.to_numpy()

    series, timestamps = to_series(linked, patient_ids)
    windows = label_windows(series[..., SERIES_CHANNELS.index("cgm_mgdl")], timestamps, cfg)
    windows.insert(0, "patient_id", patient_ids[windows.patient_idx])
    windows.insert(3, "timestamp", timestamps[windows.anchor_idx])

    splits = patient_split(ehr, cfg)
    windows["split"] = windows.patient_id.map(splits.set_index("patient_id").split)
    assert_no_patient_leakage(windows)

    raw_static = encode_static(ehr)
    train_ids = splits.patient_id[splits.split == "train"]
    encoder = StaticEncoder().fit(raw_static.loc[train_ids])
    static = encoder.transform(raw_static)

    return {"series": series, "timestamps": timestamps, "patient_ids": patient_ids, "windows": windows,
            "static": static, "encoder": encoder, "splits": splits, "link_report": report}


def summarize_dataset(ds: dict[str, Any]) -> dict[str, Any]:
    w = ds["windows"]
    out: dict[str, Any] = {"n_patients": int(len(ds["patient_ids"])), "series_shape": list(ds["series"].shape)}
    for split in SPLITS:
        s = w[w.split == split]
        noct = s[s.nocturnal]
        out[split] = {
            "patients": int(s.patient_id.nunique()),
            "windows": int(len(s)),
            "hyper_rate": round(float(s.hyper.mean()), 4),
            "hypo_rate": round(float(s.hypo.mean()), 4),
            "hypo_positives": int(s.hypo.sum()),
            "nocturnal_windows": int(len(noct)),
            "nocturnal_hypo_positives": int(noct.hypo.sum()),
            "strata": ds["splits"][ds["splits"].split == split].stratum.value_counts().to_dict(),
        }
    return out


def save_dataset(ds: dict[str, Any], cfg: dict[str, Any]) -> None:
    out = resolve_path(cfg["paths"]["processed_dir"])
    np.savez_compressed(out / "series.npz", series=ds["series"], patient_ids=ds["patient_ids"],
                        timestamps=ds["timestamps"].asi8, channels=np.array(SERIES_CHANNELS))
    ds["windows"].to_csv(out / "windows.csv.gz", index=False)
    ds["static"].to_csv(out / "static_features.csv", index_label="patient_id")
    (out / "static_encoder.json").write_text(json.dumps(ds["encoder"].to_dict(), indent=2), encoding="utf-8")
    ds["splits"].to_csv(out / "splits.csv", index=False)
    (out / "linkage_report.json").write_text(json.dumps(ds["link_report"].to_dict(), indent=2), encoding="utf-8")
    (out / "dataset_summary.json").write_text(json.dumps(summarize_dataset(ds), indent=2), encoding="utf-8")


def load_processed(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load Stage 2 outputs for model training / evaluation / the twin."""
    d = resolve_path(cfg["paths"]["processed_dir"])
    npz = np.load(d / "series.npz", allow_pickle=False)
    windows = pd.read_csv(d / "windows.csv.gz", parse_dates=["timestamp"])
    return {
        "series": npz["series"],
        "patient_ids": npz["patient_ids"],
        "timestamps": pd.DatetimeIndex(npz["timestamps"]),
        "windows": windows,
        "static": pd.read_csv(d / "static_features.csv", index_col="patient_id"),
        "encoder": StaticEncoder.from_dict(json.loads((d / "static_encoder.json").read_text(encoding="utf-8"))),
        "splits": pd.read_csv(d / "splits.csv"),
    }


def load_raw_tables(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    raw = resolve_path(cfg["paths"]["raw_dir"])
    return {
        "ehr": pd.read_csv(raw / "ehr.csv", keep_default_na=False, na_values=[""]),
        "enrollment": pd.read_csv(raw / "enrollment.csv"),
        "wearables": pd.read_csv(raw / "wearables.csv.gz", parse_dates=["timestamp"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 2 processed dataset")
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    ds = build_dataset(cfg, load_raw_tables(cfg))
    save_dataset(ds, cfg)
    print(json.dumps(ds["link_report"].to_dict() | {"ehr_without_device": len(ds["link_report"].ehr_without_device)}, indent=2, default=str))
    print(json.dumps(summarize_dataset(ds), indent=2))


if __name__ == "__main__":
    main()
