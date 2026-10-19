"""Live replay episodes, twin uncertainty bands and real-data recalibration, on small made-up data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app import replay as R
from src.eval import twin_intervals as TI
from src.eval.external_cgmacros import lopo_recalibrate

THR = {"hyper_mgdl": 180, "hypo_mgdl": 70}
T0 = pd.Timestamp("2026-01-05 00:00")


def _series() -> pd.DataFrame:
    """One day at 5 min: 120 mg/dL with a 30-min low at 10:00, a 5-min blip at 13:00 and a 30-min low at 15:00."""
    ts = pd.date_range(T0, periods=288, freq="5min")
    g = np.full(len(ts), 120.0)
    for start, mins in (("10:00", 30), ("13:00", 5), ("15:00", 30)):
        a = ts.get_loc(pd.Timestamp(f"2026-01-05 {start}"))
        g[a: a + mins // 5] = 60.0
    g[5] = np.nan                                             # a dropout must not break anything
    return pd.DataFrame({"timestamp": ts, "cgm_mgdl": g})


def _timeline(alert_times: list[str]) -> pd.DataFrame:
    ts = pd.date_range(T0 + pd.Timedelta(hours=6), T0 + pd.Timedelta(hours=22), freq="30min")
    tl = pd.DataFrame({"timestamp": ts, "anchor_idx": np.arange(len(ts)), "hypo_risk": 0.1, "hyper_risk": 0.1,
                       "hyper_alert": False, "hypo_alert": ts.isin([pd.Timestamp(f"2026-01-05 {t}") for t in alert_times])})
    tl.loc[tl.hypo_alert, "hypo_risk"] = 0.8
    s = _series().set_index("timestamp").cgm_mgdl
    tl["hypo"] = [bool((s[(s.index > t) & (s.index <= t + pd.Timedelta(hours=2))] < 70).any()) for t in ts]
    tl["hyper"] = False
    return tl


def test_onsets_need_15_minutes_past_the_threshold() -> None:
    on = R.onsets(_series(), "hypo", THR)
    assert list(on.dt.strftime("%H:%M")) == ["10:00", "15:00"]          # the 5-min blip is not an episode
    assert R.onsets(_series(), "hyper", THR).empty


def test_episodes_lead_false_and_missed() -> None:
    s, tl = _series(), _timeline(["09:00", "06:30", "12:30"])
    eps = R.episodes(s, tl, T0, T0 + pd.Timedelta(hours=23), THR)
    by_alert = {e.alert_time.strftime("%H:%M"): e for e in eps if e.alert_time is not None}
    assert by_alert["09:00"].lead_min == 60 and "60 min later" in by_alert["09:00"].sentence()
    assert by_alert["06:30"].event_time is None and "false alert" in by_alert["06:30"].sentence()
    # 12:30 warns of nothing that starts within 2 h as an episode, but the blip at 13:00 is below 70: a true alert
    assert by_alert["12:30"].event_time is None and by_alert["12:30"].true_alert
    missed = [e for e in eps if e.alert_time is None]
    assert [e.event_time.strftime("%H:%M") for e in missed] == ["15:00"] and "missed" in missed[0].sentence()


def test_replay_figure_frames_reveal_data_progressively() -> None:
    s, tl = _series(), _timeline(["09:00"])
    start, end = T0 + pd.Timedelta(hours=7), T0 + pd.Timedelta(hours=12)
    eps = R.episodes(s, tl, start, end, THR)
    fig = R.replay_figure(s, tl, start, end, {"hypo": 0.5, "hyper": 0.5}, THR, eps, step_min=10)
    assert len(fig.frames) == 31
    first, last = fig.frames[0], fig.frames[-1]
    assert len(first.data[0].x) == 1 and len(last.data[0].x) == 61         # CGM grows up to the moving line
    assert len(first.data[4].x) == 0 and len(last.data[4].x) == 1          # the hypo alert appears at 09:00
    status = [a.text for a in fig.frames[13].layout.annotations if "ALERT" in (a.text or "")]   # 09:10
    assert status and "HYPOGLYCAEMIA ALERT" in status[0]


def test_conformal_bands_reach_their_coverage() -> None:
    rng = np.random.default_rng(0)

    def draw(n: int) -> pd.DataFrame:
        lead = rng.choice([5, 10, 15], n)
        level = rng.integers(0, 3, n)
        return pd.DataFrame({"patient_id": rng.integers(0, 30, n).astype(str), "lead_min": lead, "level": level,
                             "resid": rng.normal(0, 1, n) * lead})          # error grows with lead time

    bands = TI.conformal_bands(draw(30_000))
    assert bands["lead_minutes"] == [5, 10, 15]
    lo, hi = TI.band_for(bands, 0.8, 150.0)
    assert np.all(np.diff(hi - lo) > 0)                                      # wider further ahead
    assert hi[0] == pytest.approx(1.2816 * 5, rel=0.08)
    test = draw(30_000)
    for cov in TI.COVERAGES:
        assert TI.inside(test, bands, cov).mean() == pytest.approx(cov, abs=0.015)
    assert TI.level_bin(99) == 0 and TI.level_bin(100) == 1 and TI.level_bin(181) == 2


def test_lopo_recalibration_fixes_underprediction() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    true_p = rng.uniform(0.05, 0.9, n)
    w = pd.DataFrame({"patient_id": rng.integers(0, 8, n).astype(str), "hyper": rng.uniform(size=n) < true_p,
                      "in_event_hyper": False})
    p = true_p * 0.5                                                         # the model underpredicts
    for method in ("platt", "isotonic"):
        q = lopo_recalibrate(w, p, "hyper", method)
        assert abs(q.mean() - w.hyper.mean()) < 0.03 < abs(p.mean() - w.hyper.mean())
    q = lopo_recalibrate(w, p, "hyper", "platt")
    one = (w.patient_id == "0").to_numpy()
    assert np.array_equal(np.argsort(q[one]), np.argsort(p[one]))           # ranking within a participant is kept
