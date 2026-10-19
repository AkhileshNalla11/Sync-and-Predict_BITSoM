"""Live replay: plays a stretch of one patient's record forward as if it were happening now.

The glucose trace grows every few minutes, the two risks update at each 30-min prediction time, and
alerts appear when the alert policy raised them, with the model's top reasons. Everything is a
client-side Plotly animation (no server reruns), so it plays smoothly on a cloud host and in a screen
recording. It uses only what the dashboard already has: the CGM series, the risk timeline and the
policy's alerts. SYNTHETIC DATA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.app import theme as T
from src.eval.metrics import event_episodes

NAME = {"hyper": "Hyperglycaemia", "hypo": "Hypoglycaemia"}
ICON = {"hyper": "▲", "hypo": "▼"}
ALERT_SHOWN_MIN = 90                      # an alert stays on the status card this long (the policy's refractory)
HORIZON_MIN = 120
EPISODE_MIN = 15                          # as src/eval/alert_policy.py: consensus episode, >= 15 min past threshold


@dataclass
class Episode:
    """One alert, or one glucose episode that had no alert, inside the replay window."""
    target: str
    alert_time: pd.Timestamp | None       # None: an episode with no alert in the 2 h before its onset
    risk: float
    event_time: pd.Timestamp | None       # onset of the episode this alert warned of
    reason: str = ""
    true_alert: bool = False              # the evaluation's PPV rule: any reading past the threshold within 2 h

    @property
    def lead_min(self) -> float | None:
        if self.alert_time is None or self.event_time is None:
            return None
        return (self.event_time - self.alert_time).total_seconds() / 60

    def sentence(self) -> str:
        n = NAME[self.target]
        if self.alert_time is None:
            return f"{n} episode began at {self.event_time:%H:%M} with no alert in the 2 h before it (missed)."
        head = f"{n} alert at {self.alert_time:%H:%M} (risk {self.risk:.0%})"
        if self.event_time is None:
            side = "below 70" if self.target == "hypo" else "above 180"
            if self.true_alert:
                return f"{head}: glucose went {side} within 2 h, inside an episode that had already begun (true alert)."
            return f"{head}: glucose did not go {side} in the next 2 h (false alert)."
        return f"{head}: an episode began at {self.event_time:%H:%M}, **{self.lead_min:.0f} min later**."


def onsets(series: pd.DataFrame, target: str, thresholds_mgdl: dict[str, float]) -> pd.Series:
    """Episode onsets with the alert-policy evaluation's definition (src/eval/alert_policy.py): at least
    15 min past the threshold, ended by at least 15 min clear; missing readings carry the last value."""
    dt = (series.timestamp.iloc[1] - series.timestamp.iloc[0]).total_seconds() / 60
    n = max(1, int(EPISODE_MIN // dt))
    level = thresholds_mgdl["hypo_mgdl" if target == "hypo" else "hyper_mgdl"]
    idx = event_episodes(series.cgm_mgdl.to_numpy()[None], level, below=target == "hypo",
                         min_samples=n, clear_samples=n)[0]
    return series.timestamp.iloc[idx].reset_index(drop=True)


def episodes(series: pd.DataFrame, tl: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
             thresholds_mgdl: dict[str, float], reasons: dict[tuple[str, pd.Timestamp], str] | None = None) -> list[Episode]:
    """Alerts raised in [start, end] with the episode each warned of, plus episodes in the window with no alert.

    As in the evaluation, an alert at a warns of an episode with onset e when e - 2 h <= a < e.
    """
    reasons = reasons or {}
    horizon = pd.Timedelta(minutes=HORIZON_MIN)
    out: list[Episode] = []
    for t in ("hypo", "hyper"):
        ons = onsets(series, t, thresholds_mgdl)
        all_alerts = tl[tl[f"{t}_alert"]].timestamp
        for r in tl[tl[f"{t}_alert"] & tl.timestamp.between(start, end)].itertuples():
            nxt = ons[(ons > r.timestamp) & (ons <= r.timestamp + horizon)]
            out.append(Episode(t, r.timestamp, float(getattr(r, f"{t}_risk")), nxt.iloc[0] if len(nxt) else None,
                               reasons.get((t, r.timestamp), ""), bool(getattr(r, t))))
        for on in ons[ons.between(start, end)]:
            if not ((all_alerts < on) & (all_alerts >= on - horizon)).any():
                out.append(Episode(t, None, float("nan"), on))
    return sorted(out, key=lambda e: e.alert_time if e.alert_time is not None else e.event_time)


def _status(t: pd.Timestamp, g: float, risk_row: pd.Series | None, eps: list[Episode],
            thresholds_mgdl: dict[str, float]) -> tuple[str, str, str]:
    """(html text, border colour, background) of the status card at replay time t."""
    clock = f"<b>{t:%a %H:%M}</b>"
    glucose = "no CGM reading" if not np.isfinite(g) else f"glucose <b>{g:.0f}</b> mg/dL"
    if risk_row is None:
        return f"{clock} · {glucose} · first prediction at the next half hour", T.LINE, T.CARD
    risks = f"hypo risk <b>{risk_row.hypo_risk:.0%}</b> · hyper risk <b>{risk_row.hyper_risk:.0%}</b>"
    line = f"{clock} · {glucose} · {risks}"
    shown = [e for e in eps if e.alert_time is not None
             and pd.Timedelta(0) <= t - e.alert_time <= pd.Timedelta(minutes=ALERT_SHOWN_MIN)]
    low = np.isfinite(g) and g < thresholds_mgdl["hypo_mgdl"]
    if shown:
        e = sorted(shown, key=lambda e: (e.target != "hypo", -e.alert_time.value))[0]
        mins = (t - e.alert_time).total_seconds() / 60
        head = (f"<b>{ICON[e.target]} {NAME[e.target].upper()} ALERT</b> raised {e.alert_time:%H:%M}"
                f" ({mins:.0f} min ago) · risk {e.risk:.0%}")
        if e.event_time is not None and t >= e.event_time:
            head += f" · <b>episode began {e.event_time:%H:%M}, {e.lead_min:.0f} min after the alert</b>"
        why = f"<br>Why: {e.reason}" if e.reason else ""
        return f"{line}<br>{head}{why}", T.ALERT, T.ALERT_SOFT
    if low:
        return f"{line}<br><b>▼ Glucose below 70 mg/dL</b> (the CGM's own alarm applies)", T.LOW, T.LOW_SOFT
    return line, T.LINE, T.CARD


def replay_figure(series: pd.DataFrame, tl: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                  thresholds: dict[str, float], thresholds_mgdl: dict[str, float], eps: list[Episode],
                  step_min: int = 10, frame_ms: int = 350) -> go.Figure:
    """Animated two-panel figure (glucose with alerts, risks with thresholds) with play / pause and a time slider."""
    s = series[(series.timestamp >= start) & (series.timestamp <= end)]
    r = tl[(tl.timestamp >= start) & (tl.timestamp <= end)]
    times = pd.date_range(start, end, freq=f"{step_min}min")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.66, 0.34])
    base_shapes = [
        dict(type="rect", xref="x", yref="y", x0=start, x1=end, y0=180, y1=400, fillcolor="rgba(235,104,52,0.09)",
             line_width=0, layer="below"),
        dict(type="rect", xref="x", yref="y", x0=start, x1=end, y0=40, y1=70, fillcolor="rgba(42,120,214,0.10)",
             line_width=0, layer="below"),
    ] + [dict(type="line", xref="x2", yref="y2", x0=start, x1=end, y0=thresholds[t], y1=thresholds[t],
              line=dict(color=T.TARGET_COLOR[t], width=1.2, dash="dash")) for t in ("hypo", "hyper")]
    title = dict(size=13, color=T.INK, family=T.HEAD_FONT)
    base_ann = [
        dict(x=0, y=1.0, xref="paper", yref="y domain", text="Glucose (CGM, mg/dL)", showarrow=False,
             xanchor="left", yanchor="bottom", font=title),
        dict(x=0, y=1.0, xref="paper", yref="y2 domain", text="Predicted risk for the next 2 hours", showarrow=False,
             xanchor="left", yanchor="bottom", font=title),
    ]

    def frame_parts(t: pd.Timestamp) -> tuple[list[go.Scatter], dict]:
        seen = s[s.timestamp <= t]
        rr = r[r.timestamp <= t]
        g = seen.cgm_mgdl.dropna()
        last = seen.cgm_mgdl.iloc[-1] if len(seen) else np.nan
        traces = [
            go.Scatter(x=seen.timestamp, y=seen.cgm_mgdl),
            go.Scatter(x=[seen.loc[g.index[-1], "timestamp"]] if len(g) else [], y=[g.iloc[-1]] if len(g) else []),
            go.Scatter(x=rr.timestamp, y=rr.hypo_risk),
            go.Scatter(x=rr.timestamp, y=rr.hyper_risk),
        ]
        for tgt, y in (("hypo", 47), ("hyper", 385)):
            a = rr[rr[f"{tgt}_alert"]]
            traces.append(go.Scatter(x=a.timestamp, y=np.full(len(a), y), customdata=a[f"{tgt}_risk"]))
        text, border, bg = _status(t, float(last), rr.iloc[-1] if len(rr) else None, eps, thresholds_mgdl)
        status = dict(x=0, y=1.0, yshift=30, xref="paper", yref="paper", yanchor="bottom", xanchor="left", align="left",
                      showarrow=False, text=text, bordercolor=border, borderwidth=1.5, borderpad=9, bgcolor=bg,
                      font=dict(size=13, color=T.INK, family=T.FONT))
        now_line = dict(type="line", xref="x", yref="paper", x0=t, x1=t, y0=0, y1=0.98,
                        line=dict(color=T.BRAND, width=2))
        return traces, dict(shapes=base_shapes + [now_line], annotations=base_ann + [status])

    first, layout0 = frame_parts(times[0])
    styles = [
        dict(mode="lines", name="CGM", line=dict(color=T.BRAND_DEEP, width=2.4),
             hovertemplate="%{y:.0f} mg/dL<extra>CGM</extra>", row=1),
        dict(mode="markers", name="Now", marker=dict(color=T.BRAND, size=11, line=dict(color=T.CARD, width=2)),
             hoverinfo="skip", showlegend=False, row=1),
        dict(mode="lines", name="Hypoglycaemia risk", line=dict(color=T.LOW, width=2.4),
             hovertemplate="%{y:.0%}<extra>hypo risk</extra>", row=2),
        dict(mode="lines", name="Hyperglycaemia risk", line=dict(color=T.HIGH, width=2.4),
             hovertemplate="%{y:.0%}<extra>hyper risk</extra>", row=2),
        dict(mode="markers", name="Hypo alert", marker=dict(symbol="triangle-down", size=14, color=T.LOW,
             line=dict(color=T.CARD, width=1.5)), hovertemplate="Hypo alert (risk %{customdata:.0%})<extra></extra>", row=1),
        dict(mode="markers", name="Hyper alert", marker=dict(symbol="triangle-up", size=14, color=T.HIGH,
             line=dict(color=T.CARD, width=1.5)), hovertemplate="Hyper alert (risk %{customdata:.0%})<extra></extra>", row=1),
    ]
    for tr, st_ in zip(first, styles):
        row = st_.pop("row")
        tr.update(**st_)
        fig.add_trace(tr, row=row, col=1)

    frames = []
    for t in times:
        traces, lay = frame_parts(t)
        frames.append(go.Frame(name=f"{t:%H:%M}", data=traces, traces=list(range(len(traces))), layout=lay))
    fig.frames = frames

    play = dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")
    pause = dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")
    button = dict(font=dict(size=12, color=T.INK, family=T.FONT))
    fig.update_layout(
        **layout0,
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.11, xanchor="left", yanchor="top",
                          pad=dict(r=8, t=0), showactive=False, bgcolor=T.CARD, bordercolor=T.LINE, **button,
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                   dict(label="❚❚ Pause", method="animate", args=[[None], pause])])],
        sliders=[dict(active=0, x=0.2, len=0.8, y=-0.08, yanchor="top", pad=dict(t=0, b=0),
                      currentvalue=dict(visible=False), font=dict(size=10, color=T.INK_3, family=T.FONT),
                      bordercolor=T.LINE, bgcolor=T.GRID, activebgcolor=T.BRAND, tickcolor=T.LINE,
                      steps=[dict(label=f.name if f.name.endswith(":00") else " ", method="animate",
                                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))]) for f in frames])],
    )
    fig.update_yaxes(range=[40, 400], row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", row=2, col=1)
    fig.update_xaxes(range=[start, end])
    T.style_figure(fig, height=640)
    fig.update_layout(margin=dict(l=8, r=12, t=118, b=140), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.25, x=0, xanchor="left"))
    return fig
