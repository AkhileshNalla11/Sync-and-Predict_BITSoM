"""Pair patients across the static (EHR) and dynamic (wearable) streams.

Two documented methods:

1. ``link_by_enrollment`` (used by the pipeline). The wearable stream is keyed by
   device ID; a study enrollment log maps each device to exactly one EHR patient.
   This is how a real prospective study would link the streams. In this project
   the enrollment log is written by the simulator, so **the linkage is synthetic**:
   the EHR record and the sensor trace belong to the same *virtual* patient only
   because the simulator generated them together.

2. ``link_by_attributes`` (fallback for independent sources, e.g. pairing an
   open sensor dataset such as CGMacros with synthetic EHR profiles). Greedy 1:1
   nearest-neighbour matching on standardised demographics, with exact matching
   on categorical keys. Matched pairs are *not* the same person. They are
   statistically plausible stand-ins, and every row is flagged as such.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


class LinkageError(ValueError):
    """Raised when the crosswalk is not a clean 1:1 mapping."""


@dataclass
class LinkReport:
    method: str
    n_ehr_patients: int
    n_devices_with_data: int
    n_linked: int
    ehr_without_device: list[str]
    devices_without_enrollment: list[str]
    synthetic_linkage: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def link_by_enrollment(ehr: pd.DataFrame, wearables: pd.DataFrame,
                       enrollment: pd.DataFrame) -> tuple[pd.DataFrame, LinkReport]:
    """Attach ``patient_id`` to every wearable row via the enrollment crosswalk.

    Validates that the crosswalk is 1:1. Devices with no enrollment record are
    dropped (reported); patients with no device are reported.
    """
    for col in ("patient_id", "device_id"):
        dup = enrollment[col][enrollment[col].duplicated()]
        if len(dup):
            raise LinkageError(f"enrollment is not 1:1: duplicated {col} {sorted(dup.unique())[:5]}")
    unknown = sorted(set(enrollment.patient_id) - set(ehr.patient_id))
    if unknown:
        raise LinkageError(f"enrollment references patients missing from the EHR: {unknown[:5]}")

    devices = set(wearables.device_id.unique())
    orphan_devices = sorted(devices - set(enrollment.device_id))
    linked = wearables.merge(enrollment[["device_id", "patient_id"]], on="device_id", how="inner", validate="many_to_one")
    linked_patients = set(linked.patient_id.unique())
    report = LinkReport(
        method="enrollment_crosswalk",
        n_ehr_patients=int(ehr.patient_id.nunique()),
        n_devices_with_data=len(devices),
        n_linked=len(linked_patients),
        ehr_without_device=sorted(set(ehr.patient_id) - linked_patients),
        devices_without_enrollment=orphan_devices,
    )
    cols = ["patient_id"] + [c for c in linked.columns if c != "patient_id"]
    return linked[cols], report


def link_by_attributes(ehr: pd.DataFrame, devices: pd.DataFrame, numeric: list[str],
                       exact: list[str]) -> pd.DataFrame:
    """Greedy 1:1 nearest-neighbour pairing of device participants to EHR profiles.

    Args:
        ehr:     one row per EHR patient (``patient_id`` + attribute columns).
        devices: one row per device participant (``device_id`` + the same attributes).
        numeric: columns matched by standardised Euclidean distance (e.g. age, bmi).
        exact:   columns that must match exactly (e.g. sex, diabetes status).

    Returns a linkage table (device_id, patient_id, distance, method). Devices
    with no admissible partner left are omitted. Closest pairs are matched first,
    so the result does not depend on input row order.
    """
    scale = ehr[numeric].std(ddof=0).replace(0, 1.0)
    e = (ehr[numeric] / scale).to_numpy()
    d = (devices[numeric] / scale).to_numpy()
    dist = np.sqrt(((d[:, None, :] - e[None, :, :]) ** 2).sum(-1))
    for col in exact:
        dist[devices[col].to_numpy()[:, None] != ehr[col].to_numpy()[None, :]] = np.inf

    pairs, used_dev, used_ehr = [], set(), set()
    for flat in np.argsort(dist, axis=None, kind="stable"):
        i, j = divmod(int(flat), dist.shape[1])
        if not np.isfinite(dist[i, j]):
            break
        if i in used_dev or j in used_ehr:
            continue
        used_dev.add(i)
        used_ehr.add(j)
        pairs.append((devices.device_id.iloc[i], ehr.patient_id.iloc[j], float(dist[i, j])))
    out = pd.DataFrame(pairs, columns=["device_id", "patient_id", "distance"])
    out["method"] = "attribute_match_SYNTHETIC_not_same_person"
    return out
