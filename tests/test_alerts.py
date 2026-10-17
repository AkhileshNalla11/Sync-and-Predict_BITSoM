"""Alert policy logic (src/twin/alerts.py) and the episode-level metrics behind docs/alert_policy.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.alert_policy import Split, episode_leads, per_patient, select, summarise
from src.eval.metrics import event_episodes
from src.twin.alerts import AlertPolicy, policy_alerts, policy_from_spec, run_policy

STEP = 30                                    # minutes between predictions


def _fire(risk, policy, in_event=None, day=None) -> list[int]:
    t = np.arange(len(risk)) * STEP
    return np.flatnonzero(run_policy(t, np.asarray(risk, float), policy, in_event, day)).tolist()


# --------------------------------------------------------------------------- policy
def test_window_level_alerts_every_window_above_threshold() -> None:
    risk = [0.1, 0.5, 0.6, 0.2, 0.7]
    assert _fire(risk, AlertPolicy.window_level(0.5)) == [1, 2, 4]


def test_refractory_suppresses_repeats_then_reminds() -> None:
    risk = [0.9] * 10                                         # risk stays high for 5 h
    assert _fire(risk, AlertPolicy(0.5, refractory_min=120)) == [0, 4, 8]       # every 2 h at most
    assert _fire(risk, AlertPolicy(0.5, refractory_min=0)) == list(range(10))


def test_refractory_counts_real_time_across_gaps() -> None:
    t = np.array([0, 30, 60, 300, 330])                       # a 4 h gap in the predictions
    fired = run_policy(t, np.full(5, 0.9), AlertPolicy(0.5, refractory_min=120))
    assert np.flatnonzero(fired).tolist() == [0, 3]


def test_hysteresis_needs_risk_to_fall_below_rearm_level() -> None:
    pol = AlertPolicy(0.5, rearm=0.25)
    # dips to 0.4 (below threshold, above re-arm) do not re-arm; the dip to 0.2 does
    risk = [0.6, 0.4, 0.6, 0.4, 0.6, 0.2, 0.6]
    assert _fire(risk, pol) == [0, 6]
    # re-arm == threshold: edge-triggered, one alert per crossing
    assert _fire(risk, AlertPolicy(0.5, rearm=0.5)) == [0, 2, 4, 6]


def test_hysteresis_and_refractory_both_required() -> None:
    risk = [0.6, 0.1, 0.6, 0.6, 0.6, 0.6]                     # re-armed at once, but refractory 2 h
    assert _fire(risk, AlertPolicy(0.5, rearm=0.25, refractory_min=120)) == [0, 4]


def test_escalation_realerts_without_rearming() -> None:
    risk = [0.1, 0.2, 0.25, 0.3, 0.5, 0.65, 0.9, 0.95]
    pol = AlertPolicy(0.2, rearm=0.1, refractory_min=60, escalation=3.0)
    # 0.2 -> needs >= 0.6 (and 60 min after the last alert) -> 0.65; next would need >= 1.95
    assert _fire(risk, pol) == [1, 5]
    assert _fire(risk, AlertPolicy(0.2, rearm=0.1, refractory_min=60)) == [1]


def test_daily_budget_caps_alerts_per_calendar_day() -> None:
    risk = np.full(8, 0.9)
    day = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert _fire(risk, AlertPolicy(0.5, daily_budget=2), day=day) == [0, 1, 4, 5]
    with pytest.raises(ValueError):
        _fire(risk, AlertPolicy(0.5, daily_budget=2))         # budget needs the day labels


def test_no_predictive_alert_while_already_in_event() -> None:
    risk = [0.9, 0.9, 0.9]
    in_event = np.array([True, False, False])
    assert _fire(risk, AlertPolicy(0.5, rearm=0.25), in_event) == [1]
    assert _fire(risk, AlertPolicy(0.5, suppress_in_event=False), in_event) == [0, 1, 2]


def test_invalid_policies_rejected() -> None:
    with pytest.raises(ValueError):
        AlertPolicy(0.5, rearm=0.6)
    with pytest.raises(ValueError):
        AlertPolicy(0.5, escalation=1.0)
    with pytest.raises(ValueError):
        run_policy(np.array([0, 0]), np.array([0.9, 0.9]), AlertPolicy(0.5))     # times must increase


def test_policy_from_spec() -> None:
    pol = policy_from_spec({"rearm_fraction": 0.5, "refractory_min": 120, "daily_budget": None, "escalation": 2}, 0.08)
    assert pol == AlertPolicy(0.08, rearm=0.04, refractory_min=120, daily_budget=None, escalation=2.0)
    assert policy_from_spec({"rearm_fraction": None}, 0.08).rearm is None


def test_policy_alerts_runs_per_patient_in_any_row_order() -> None:
    ts = pd.date_range("2026-01-05 08:00", periods=4, freq="30min")
    w = pd.DataFrame({"patient_id": ["A"] * 4 + ["B"] * 4, "timestamp": list(ts) * 2, "in_event_hypo": False})
    risk = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9])
    pol = AlertPolicy(0.5, refractory_min=120)
    expected = np.array([1, 0, 0, 0, 0, 1, 0, 0], bool)       # B's first alert is not blocked by A's
    np.testing.assert_array_equal(policy_alerts(w, risk, pol, "hypo"), expected)
    perm = np.random.default_rng(0).permutation(len(w))
    np.testing.assert_array_equal(policy_alerts(w.iloc[perm], risk[perm], pol, "hypo"), expected[perm])


# --------------------------------------------------------------------------- episodes and metrics
def test_consensus_episodes_need_15_min_in_and_out() -> None:
    x = np.full(40, 100.0)
    x[5:7] = 60                  # 10 min below: not an episode
    x[10:14] = 60                # 20 min below: episode at 10
    x[14:16] = 75                # 10 min recovery: same episode continues
    x[16:18] = 60
    x[18:22] = 80                # 20 min recovery: episode over
    x[25:28] = 65                # new episode at 25
    x[30] = np.nan               # missing reading carries 100 forward
    ons = event_episodes(x[None], 70, below=True, min_samples=3, clear_samples=3)
    assert ons[0].tolist() == [10, 25]


def _split(alert_anchors: list[int], onsets: list[int], labels: list[bool] | None = None) -> tuple[Split, np.ndarray]:
    anchors = np.arange(0, 120, 6)                            # predictions every 30 min (6 samples)
    w = pd.DataFrame({"patient_idx": 0, "anchor_idx": anchors,
                      "hypo": labels if labels is not None else False, "nocturnal": False})
    ep = pd.DataFrame({"patient_idx": 0, "onset_idx": onsets, "nocturnal": False})
    split = Split(windows=w, risk={}, episodes={"hypo": ep}, stride_h=0.5, horizon=24, dt=5)
    return split, np.isin(anchors, alert_anchors)


def test_episode_lead_uses_earliest_alert_within_horizon() -> None:
    split, alerts = _split([30, 42, 66], [60, 100])
    L = episode_leads(split, alerts, "hypo")
    # onset 60: alerts at 42 (lead 90 min) qualify; 30 is > 2 h early. onset 100: alert 66 is 170 min early -> missed
    assert L.lead_min.iloc[0] == 90 and np.isnan(L.lead_min.iloc[1])
    assert L.eligible.all()


def test_summary_counts_alerts_ppv_and_catch_rates() -> None:
    labels = [False] * 20
    labels[7] = True                                          # anchor 42's horizon contains a low
    split, alerts = _split([42, 90], [60], labels)
    s = summarise(per_patient(split, alerts, "hypo"), split)
    assert s["episodes"] == 1 and s["caught_30"] == 1.0
    assert s["alerts_per_day"] == pytest.approx(2 / (20 * 0.5 / 24))
    assert s["ppv"] == 0.5 and s["false_alerts_per_day"] == pytest.approx(1 / (20 * 0.5 / 24))


def test_selection_respects_the_alert_cap() -> None:
    base = {"kind": "policy", "rearm_fraction": np.nan, "daily_budget": np.nan, "escalation": np.nan, "refractory_min": 120}
    sw = pd.DataFrame([
        {**base, "target": "hypo", "caught_30": 0.99, "alerts_per_day": 3.0},   # best catch, over the cap
        {**base, "target": "hypo", "caught_30": 0.90, "alerts_per_day": 0.9},
        {**base, "target": "hypo", "caught_30": 0.895, "alerts_per_day": 0.5},  # within 1 point, fewer alerts
        {**base, "target": "hyper", "caught_30": 0.5, "alerts_per_day": 1.0},
    ])
    chosen = select(sw)
    assert chosen["hypo"].alerts_per_day == 0.5 and chosen["hyper"].caught_30 == 0.5
