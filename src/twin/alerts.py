"""Alert policy: turn a stream of 30-minute risk predictions into alerts a clinician could live with.

Scoring every window against a threshold raises the same alarm again and again while risk stays high
(about 5-8 alerting windows per patient-day). The policy fires once when risk crosses the threshold,
then stays quiet until both:

  * the refractory period has passed since the last alert (e.g. 2 h), and
  * risk has dropped below a lower re-arm level (hysteresis), so a risk that hovers around the
    threshold does not alert on every wiggle,

unless risk ESCALATES: once it reaches `escalation` x the risk at the last alert (e.g. 3x: 8% -> 24%),
one more alert may fire after the refractory period even without re-arming. Slow drifts often cross
the threshold hours before the low; escalation re-warns when the low is actually close.

Optionally it caps alerts per patient per calendar day (budget) and skips windows where the CGM
already reads past the threshold (the CGM's own threshold alarm covers those; a *predictive* alert
there adds nothing). Hypo and hyper each get their own policy.

    pol = AlertPolicy(threshold=0.06, rearm=0.03, refractory_min=120, escalation=3.0)
    fired = policy_alerts(windows, risk, pol, "hypo")      # bool per window, aligned with `windows`

The deployed setting (config.yaml `twin.alert_policy`, chosen on validation patients) uses only the
threshold and the refractory period: on the simulator's slow drifts, hysteresis kept the policy silent
when the low arrived (docs/alert_policy.md).

Window-level alerting (the old behaviour) is the special case
AlertPolicy(threshold, rearm=None, refractory_min=0, suppress_in_event=False).
SYNTHETIC DATA, NOT FOR CLINICAL USE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TARGETS = ("hyper", "hypo")


@dataclass(frozen=True)
class AlertPolicy:
    threshold: float                     # alert when risk >= threshold (and the policy is armed)
    rearm: float | None = None           # re-arm only once risk < rearm; None = no hysteresis (re-armed at once)
    refractory_min: float = 0.0          # no new alert within this many minutes of the previous one
    daily_budget: int | None = None      # at most this many alerts per patient per calendar day; None = no cap
    escalation: float | None = None      # while disarmed, re-alert once risk >= escalation x risk at the last alert
    suppress_in_event: bool = True       # no predictive alert while the CGM is already past the threshold

    def __post_init__(self) -> None:
        if self.rearm is not None and self.rearm > self.threshold:
            raise ValueError(f"re-arm level {self.rearm} must not exceed the alert threshold {self.threshold}")
        if self.refractory_min < 0:
            raise ValueError("refractory_min must be >= 0")
        if self.daily_budget is not None and self.daily_budget < 1:
            raise ValueError("daily_budget must be >= 1 (or None for no cap)")
        if self.escalation is not None and self.escalation <= 1:
            raise ValueError("escalation must be > 1 (or None)")

    @classmethod
    def window_level(cls, threshold: float) -> "AlertPolicy":
        """The pre-policy behaviour: every window at or above the threshold is an alert."""
        return cls(threshold, rearm=None, refractory_min=0.0, daily_budget=None, suppress_in_event=False)

    def describe(self) -> str:
        """Plain-language summary, e.g. 'alert at risk >= 5.5%, then at most one alert per 2 h'."""
        parts = [f"alert at risk >= {self.threshold:.1%}"]
        if self.refractory_min:
            parts.append(f"then at most one alert per {self.refractory_min / 60:g} h")
        if self.rearm is not None:
            parts.append(f"re-arms only after risk falls below {self.rearm:.1%}")
        if self.escalation is not None:
            parts.append(f"or if risk rises to {self.escalation:g}x its level at the last alert")
        if self.daily_budget is not None:
            parts.append(f"max {self.daily_budget} per day")
        if self.suppress_in_event:
            parts.append("none while the CGM already reads past the threshold")
        return ", ".join(parts)


def run_policy(minutes: np.ndarray, risk: np.ndarray, policy: AlertPolicy, in_event: np.ndarray | None = None,
               day: np.ndarray | None = None) -> np.ndarray:
    """Alerts for ONE patient's time-ordered predictions. Returns a bool array (True = alert fired here).

    `minutes`: prediction times in minutes (increasing; gaps allowed). `in_event`: CGM already past the
    threshold at that time. `day`: calendar-day label per prediction (needed only for the budget).
    """
    minutes = np.asarray(minutes, float)
    risk = np.asarray(risk, float)
    if len(minutes) != len(risk):
        raise ValueError("minutes and risk must have the same length")
    if len(minutes) > 1 and np.any(np.diff(minutes) <= 0):
        raise ValueError("prediction times must be strictly increasing")
    if policy.daily_budget is not None and day is None:
        raise ValueError("a daily budget needs the calendar day of each prediction")
    fired = np.zeros(len(risk), bool)
    if policy.rearm is None and policy.refractory_min == 0 and policy.daily_budget is None:   # stateless: vectorise
        fired[:] = risk >= policy.threshold
        if in_event is not None and policy.suppress_in_event:
            fired &= ~np.asarray(in_event, bool)
        return fired
    skip = [False] * len(risk) if in_event is None or not policy.suppress_in_event else np.asarray(in_event, bool).tolist()
    minutes, risk = minutes.tolist(), risk.tolist()          # plain floats: much faster in the loop below
    esc = policy.escalation
    armed, last, last_risk = True, -np.inf, np.inf
    per_day: dict[Any, int] = {}
    for i in range(len(risk)):
        r = risk[i]
        if not armed and policy.rearm is not None and r < policy.rearm:
            armed = True
        if r < policy.threshold or skip[i]:
            continue
        if policy.rearm is not None and not armed and not (esc is not None and r >= esc * last_risk):
            continue
        if minutes[i] - last < policy.refractory_min:
            continue
        if policy.daily_budget is not None and per_day.get(day[i], 0) >= policy.daily_budget:
            continue
        fired[i] = True
        last, last_risk = minutes[i], r
        armed = False
        if policy.daily_budget is not None:
            per_day[day[i]] = per_day.get(day[i], 0) + 1
    return fired


def policy_alerts(windows: pd.DataFrame, risk: np.ndarray, policy: AlertPolicy, target: str) -> np.ndarray:
    """Apply `policy` per patient to prediction windows in any row order; result is aligned with `windows`.

    Needs columns patient_id, timestamp and in_event_<target>.
    """
    ts = pd.to_datetime(windows.timestamp).to_numpy()
    minutes = (ts - ts.min()).astype("timedelta64[s]").astype(np.int64) / 60.0 if len(ts) else np.zeros(0)
    day = pd.to_datetime(windows.timestamp).dt.normalize().to_numpy()
    in_event = windows[f"in_event_{target}"].to_numpy(bool)
    risk = np.asarray(risk, float)
    out = np.zeros(len(windows), bool)
    codes = pd.factorize(windows.patient_id)[0]
    order = np.lexsort((minutes, codes))
    pid = codes[order]
    bounds = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        out[rows] = run_policy(minutes[rows], risk[rows], policy, in_event[rows], day[rows])
    return out


# --------------------------------------------------------------------------- configuration
def policy_from_spec(spec: dict[str, Any], threshold: float) -> AlertPolicy:
    """Build a policy from a config entry (see config.yaml `twin.alert_policy`) and its arm threshold."""
    frac = spec.get("rearm_fraction")
    budget = spec.get("daily_budget")
    esc = spec.get("escalation")
    return AlertPolicy(threshold=float(threshold), rearm=None if frac is None else float(frac) * float(threshold),
                       refractory_min=float(spec.get("refractory_min", 0)),
                       daily_budget=None if budget is None else int(budget),
                       escalation=None if esc is None else float(esc),
                       suppress_in_event=bool(spec.get("suppress_in_event", True)))


def policy_far(cfg: dict[str, Any]) -> dict[str, float]:
    """Per-target false-alert rate (on validation onset windows) that sets each policy's arm threshold."""
    return {t: float(cfg["twin"]["alert_policy"][t]["false_alert_rate"]) for t in TARGETS}
