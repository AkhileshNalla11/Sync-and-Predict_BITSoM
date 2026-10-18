"""Subgroup analysis: grouping, suppression of small groups, metric correctness."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.eval.metrics import auroc
from src.eval.subgroups import MIN_POSITIVES, subgroup_labels, subgroup_table


def _ehr(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"patient_id": [f"P{i}" for i in range(n)], "sex": rng.choice(["M", "F"], n),
                         "age": rng.integers(35, 80, n), "egfr": rng.integers(40, 110, n),
                         "on_insulin": rng.random(n) < 0.2, "on_sulfonylurea": rng.random(n) < 0.4,
                         "hba1c": rng.uniform(6, 10, n).round(1), "bmi": rng.uniform(20, 32, n)})


def test_labels_are_aligned_to_patients() -> None:
    ehr = _ehr(40)
    g = subgroup_labels(ehr)
    assert list(g.index) == list(ehr.patient_id)
    assert not g.isna().any().any()
    assert (g.loc[ehr.patient_id[ehr.age >= 65], "Age"] == ">= 65").all()
    assert (g.loc[ehr.patient_id[ehr.on_insulin], "Drug class"] == "Insulin").all()


def test_table_scores_big_groups_and_suppresses_small_ones() -> None:
    cfg = load_config()
    cfg["eval"]["n_bootstrap"] = 20
    ehr = _ehr(40)
    rng = np.random.default_rng(1)
    pid = np.repeat(ehr.patient_id.to_numpy(), 50)
    y = (rng.random(len(pid)) < 0.2).astype(int)
    w = pd.DataFrame({"patient_id": pid, "hyper": y, "hypo": np.zeros_like(y), "nocturnal": False,
                      "in_event_hyper": False, "in_event_hypo": False})
    s = y + rng.normal(0, 0.7, len(y))
    out = subgroup_table(w, {"hyper": s, "hypo": rng.random(len(y))}, subgroup_labels(ehr), cfg)
    hyper = out[out.target == "hyper"]
    assert hyper.reported.all()
    male = hyper[(hyper.dimension == "Sex") & (hyper.group == "Male")].iloc[0]
    m = np.isin(pid, ehr.patient_id[ehr.sex == "M"])
    assert male.auroc == auroc(y[m], s[m]) and male.auroc_lo <= male.auroc <= male.auroc_hi
    assert not out[out.target == "hypo"].reported.any()          # no hypo events at all -> nothing scored
    assert (out[out.target == "hypo"].positives < MIN_POSITIVES).all()
