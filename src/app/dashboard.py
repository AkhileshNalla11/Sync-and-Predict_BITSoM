"""Clinician dashboard for the glycaemic digital twin (Streamlit).

    streamlit run src/app/dashboard.py        # or: scripts\\demo.cmd / bash scripts/demo.sh

Views: live replay, glucose timeline with both risks and alerts, alert drivers, what-if panel, patient overview.
Alerts follow the alert policy (src/twin/alerts.py, docs/alert_policy.md): one alert per risk episode.
SYNTHETIC DATA. Research prototype, NOT FOR CLINICAL USE. Styling lives in src/app/theme.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                      # `streamlit run` puts only this folder on the path
    sys.path.insert(0, str(ROOT))

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
import plotly.graph_objects as go                   # noqa: E402
import streamlit as st                              # noqa: E402
from plotly.subplots import make_subplots           # noqa: E402

from src.app import replay as R                     # noqa: E402
from src.app import theme as T                      # noqa: E402
from src.config import load_config                  # noqa: E402
from src.eval.twin_intervals import band_for        # noqa: E402
from src.twin.explain import driver_sentences, explain, group_contributions   # noqa: E402
from src.twin.alerts import policy_far              # noqa: E402
from src.twin.state import PatientTwin, TwinContext  # noqa: E402
from src.twin.personalise import FIT_PARAMS        # noqa: E402
from src.twin.whatif import (MAIN_MEALS, MEAL_CATALOGUE, AddActivity, AddMeal, DelayMeal, ExtraDose,  # noqa: E402
                             PatientSimulator, PoorSleep, Scenario, ScaleMedication, SkipMeal, WhatIfResult,
                             run_whatif)

TARGET_NAME = {"hyper": "Hyperglycaemia", "hypo": "Hypoglycaemia"}
TARGET_RULE = {"hyper": "Any glucose > 180 mg/dL in the next 2 h", "hypo": "Any glucose < 70 mg/dL in the next 2 h"}
HIGH_BAND, LOW_BAND = "rgba(235,104,52,0.09)", "rgba(42,120,214,0.10)"
REAL_MODEL = "lgbm_cgm_shanghai"
RISK_MODELS = {"Synthetic-trained: CGM + wearables + EHR": None,
               "Real-patient-trained: CGM only (100 patients, ShanghaiT2DM)": REAL_MODEL}
TWIN_MODES = {"Personalised (fitted to this patient's data)": "fitted",
              "Oracle (hidden simulator truth, for comparison only)": "oracle"}
PARAM_LABEL = {"g_basal": "Basal glucose", "meal_gain": "Carb sensitivity", "tau_clear_min": "Clearance time",
               "drug_sens": "Drug sensitivity"}


# --------------------------------------------------------------------------- cached resources
def runtime_config() -> dict:
    """Use the full pipeline outputs if present, else the committed demo bundle (e.g. on a cloud host)."""
    cfg = load_config()
    local = ROOT / cfg["paths"]["processed_dir"] / "series.npz"
    bundle = ROOT / "demo_bundle" / "processed" / "series.npz"
    if os.environ.get("TWIN_USE_DEMO_BUNDLE") == "1" or (not local.exists() and bundle.exists()):
        cfg["paths"] = {"raw_dir": "demo_bundle/raw", "processed_dir": "demo_bundle/processed",
                        "models_dir": "demo_bundle/processed/models", "docs_dir": cfg["paths"]["docs_dir"]}
    return cfg


@st.cache_resource(show_spinner="Loading the twin (data, model, thresholds)…")
def get_context() -> TwinContext:
    return TwinContext.load(runtime_config())


@st.cache_resource(show_spinner="Loading the risk model…")
def get_model_context(model: str) -> TwinContext:
    """The twin with a given risk model; shares the loaded data with get_context()."""
    return get_context().with_model(model)


def real_model_info(ctx: TwinContext) -> dict | None:
    """Training provenance and real-patient alert results saved with the real-trained model (None otherwise)."""
    path = ROOT / ctx.cfg["paths"]["models_dir"] / f"{REAL_MODEL}_alert_thresholds.json"
    return json.loads(path.read_text(encoding="utf-8")) if ctx.model_name == REAL_MODEL and path.exists() else None


@st.cache_resource(show_spinner="Building this patient's twin (a few seconds if the fit is not cached)…", max_entries=8)
def get_simulator(patient_id: str, mode: str = "fitted") -> PatientSimulator:
    return PatientSimulator(get_context().cfg, patient_id, mode=mode)


@st.cache_data(show_spinner=False, max_entries=32)
def get_timeline(patient_id: str, model: str) -> pd.DataFrame:
    return get_model_context(model).risk_timeline(patient_id)


@st.cache_data(show_spinner=False, max_entries=32)
def get_series(patient_id: str) -> pd.DataFrame:
    return get_context().series_frame(patient_id)


@st.cache_data(show_spinner=False)
def patient_table() -> pd.DataFrame:
    """Sidebar patient list. Built from the labels alone (no model calls) so the first page load stays fast."""
    ctx = get_context()
    ids = ctx.patients("test")
    hypo_events = ctx.windows[ctx.windows.patient_id.isin(ids)].groupby("patient_id").hypo.sum()
    e = ctx.ehr.loc[ids]
    return pd.DataFrame({"patient_id": ids, "regimen": e.regimen.to_numpy(), "hba1c": e.hba1c.to_numpy(),
                         "age": e.age.to_numpy(), "sex": e.sex.to_numpy(),
                         "hypo_events": hypo_events.reindex(ids).fillna(0).astype(int).to_numpy()})


@st.cache_data(show_spinner=False)
def get_intervals() -> dict | None:
    """Calibrated uncertainty bands for the twin's projection (src/eval/twin_intervals.py), if computed."""
    path = ROOT / runtime_config()["paths"]["models_dir"] / "twin_intervals.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False, max_entries=16)
def alert_reasons(patient_id: str, start: pd.Timestamp, end: pd.Timestamp, model: str) -> dict[tuple[str, pd.Timestamp], str]:
    """Top two risk-raising drivers of every alert in the window, for the live replay's status card."""
    ctx, tl = get_model_context(model), get_timeline(patient_id, model)
    out = {}
    for t in ("hypo", "hyper"):
        for r in tl[tl[f"{t}_alert"] & tl.timestamp.between(start, end)].itertuples():
            d = explain(ctx, ctx.build_twin(patient_id, int(r.anchor_idx)), t, top_k=8)
            out[(t, r.timestamp)] = " · ".join(f"{x.label}: {x.value}" for x in d[d.contribution > 0].head(2).itertuples())
    return out


def cached_whatif(ctx: TwinContext, sim: PatientSimulator, patient_id: str, anchor_idx: int, mode: str,
                  scenarios: list) -> WhatIfResult:
    """Re-simulate only when the patient, time, twin mode or scenario changes (results kept per browser session)."""
    store = st.session_state.setdefault("whatif_cache", {})
    key = (ctx.model_name, patient_id, int(anchor_idx), mode, tuple(repr(sc) for sc in scenarios))
    if key not in store:
        if len(store) >= 24:
            store.pop(next(iter(store)))
        with st.spinner("Re-simulating and re-predicting…"):
            store[key] = run_whatif(ctx, sim, anchor_idx, scenarios)
    return store[key]


# --------------------------------------------------------------------------- helpers
def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def fmt_time(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%a %d %b · %H:%M")


def regimen_label(regimen: str) -> str:
    return regimen.replace("+", " + ").replace("sulfonylurea", "sulfonylurea").capitalize()


def alert_state(tl: pd.DataFrame, anchor_idx: int, target: str) -> tuple[str, str]:
    """Alert-policy state at this prediction time: (alert | active | ok, one-line explanation)."""
    upto = tl[tl.anchor_idx <= anchor_idx]
    row = upto.iloc[-1]
    prev = upto[upto[f"{target}_alert"]]
    last = f"last alert {fmt_time(prev.timestamp.iloc[-1])}" if len(prev) else "no earlier alert"
    if row[f"{target}_alert"]:
        return "alert", "The alert policy raises an alert at this prediction time."
    if row[f"{target}_above"]:
        if row[f"in_event_{target}"]:
            return "active", "CGM already reads past the threshold: the CGM's own alarm applies, no predictive alert."
        return "active", f"Repeat suppressed by the alert policy ({last})."
    return "ok", f"No alert now ({last})."


def far_label(ctx: TwinContext, target: str) -> str:
    where = "real patients (ShanghaiT2DM)" if ctx.model_name == REAL_MODEL else "validation patients"
    return f"{policy_far(ctx.cfg)[target] * 100:g}% false-alert rate on {where}"


def cgm_summary(series: pd.DataFrame) -> dict[str, float]:
    g = series.cgm_mgdl.dropna()
    return {"mean": g.mean(), "gmi": 3.31 + 0.02392 * g.mean(), "tir": ((g >= 70) & (g <= 180)).mean(),
            "tar": (g > 180).mean(), "tbr": (g < 70).mean(), "cv": g.std() / g.mean(),
            "coverage": series.cgm_mgdl.notna().mean()}


# --------------------------------------------------------------------------- figures
def timeline_figure(series: pd.DataFrame, tl: pd.DataFrame, now: pd.Timestamp, hours_back: int,
                    thresholds: dict[str, float]) -> go.Figure:
    start, end = now - pd.Timedelta(hours=hours_back), now + pd.Timedelta(hours=2)
    s = series[(series.timestamp >= start) & (series.timestamp <= end)]
    past, future = s[s.timestamp <= now], s[s.timestamp >= now]
    r = tl[(tl.timestamp >= start) & (tl.timestamp <= now)]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.56, 0.29, 0.15],
                        subplot_titles=("Glucose (CGM, mg/dL)", "Predicted risk for the next 2 hours", "Activity and sleep"))
    for y0, y1, color in ((180, 400, HIGH_BAND), (40, 70, LOW_BAND)):
        fig.add_shape(type="rect", xref="x", yref="y", x0=start, x1=end, y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")
    fig.add_annotation(x=start, y=192, xref="x", yref="y", text="HIGH  > 180", showarrow=False, xanchor="left",
                       font=dict(size=10, color=T.HIGH, family=T.FONT))
    fig.add_annotation(x=start, y=62, xref="x", yref="y", text="LOW  < 70", showarrow=False, xanchor="left",
                       font=dict(size=10, color=T.LOW, family=T.FONT))
    fig.add_trace(go.Scatter(x=past.timestamp, y=past.cgm_mgdl, mode="lines", name="CGM (seen by model)",
                             line=dict(color=T.BRAND_DEEP, width=2.4), hovertemplate="%{y:.0f} mg/dL<extra>CGM</extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=future.timestamp, y=future.cgm_mgdl, mode="lines", name="Actual next 2 h (not seen by model)",
                             line=dict(color=T.INK_3, width=2.2, dash="dot"), hovertemplate="%{y:.0f} mg/dL<extra>actual</extra>"), 1, 1)

    for t, sym, y in (("hyper", "triangle-up", 385), ("hypo", "triangle-down", 47)):
        a = r[r[f"{t}_alert"]]
        quiet = r[r[f"{t}_above"] & ~r[f"{t}_alert"]]
        color = T.TARGET_COLOR[t]
        fig.add_trace(go.Scatter(x=quiet.timestamp, y=np.full(len(quiet), y), mode="markers",
                                 name=f"{TARGET_NAME[t]}: above threshold, repeat suppressed",
                                 marker=dict(symbol=f"{sym}-open", size=8, color=color, line=dict(width=1.2)),
                                 customdata=quiet[f"{t}_risk"],
                                 hovertemplate="Above threshold (risk %{customdata:.0%}), no new alert: "
                                               "suppressed by the alert policy<extra></extra>"), 1, 1)
        fig.add_trace(go.Scatter(x=a.timestamp, y=np.full(len(a), y), mode="markers", name=f"{TARGET_NAME[t]} alert raised",
                                 marker=dict(symbol=sym, size=12, color=color, line=dict(color=T.CARD, width=1.5)),
                                 customdata=a[f"{t}_risk"],
                                 hovertemplate=TARGET_NAME[t] + " alert raised (risk %{customdata:.0%})<extra></extra>"), 1, 1)
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=r.timestamp, y=r[f"{t}_risk"], mode="lines", name=f"{TARGET_NAME[t]} risk",
                                 line=dict(color=color, width=2.4, shape="spline", smoothing=0.4),
                                 fill="tozeroy", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.10)",
                                 hovertemplate="%{y:.0%}<extra>" + TARGET_NAME[t] + " risk</extra>", showlegend=False), 2, 1)
        fig.add_hline(y=thresholds[t], line=dict(color=color, width=1.2, dash="dash"), row=2, col=1)

    fig.add_trace(go.Bar(x=past.timestamp, y=past.steps, name="Steps / 5 min", marker_color="#b9c5d3",
                         hovertemplate="%{y:.0f} steps<extra></extra>", showlegend=False), 3, 1)
    asleep = past[past.sleep_stage.isin(["light", "deep", "rem"])]
    fig.add_trace(go.Scatter(x=asleep.timestamp, y=np.full(len(asleep), 520), mode="markers", name="Asleep",
                             marker=dict(symbol="square", size=6, color="#6e59c9", opacity=0.55),
                             hovertemplate="asleep (%{text})<extra></extra>", text=asleep.sleep_stage, showlegend=False), 3, 1)

    for row in (1, 2, 3):
        fig.add_vline(x=now, line=dict(color=T.BRAND, width=2), row=row, col=1)
    fig.add_annotation(x=now, y=1.0, yref="y domain", text=" NOW ", showarrow=False, yanchor="bottom",
                       font=dict(size=10, color="#ffffff", family=T.FONT), bgcolor=T.BRAND, borderpad=3)
    fig.update_yaxes(range=[40, 400], row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", row=2, col=1)
    fig.update_yaxes(range=[0, 560], showticklabels=False, row=3, col=1)
    fig.update_xaxes(range=[start, end])
    T.style_figure(fig, height=700)
    # Six legend entries: below the plot they have room and never collide with the panel titles.
    fig.update_layout(bargap=0.1, margin=dict(l=8, r=12, t=36, b=86),
                      legend=dict(orientation="h", yanchor="top", y=-0.07, x=0, xanchor="left"))
    for a in fig.layout.annotations[:3]:
        a.update(font=dict(size=13, color=T.INK, family=T.HEAD_FONT), x=0, xanchor="left")
    return fig


def drivers_figure(drivers: pd.DataFrame) -> go.Figure:
    d = drivers.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d.contribution, y=[f"{r.label}: <b>{r.value}</b>" for r in d.itertuples()], orientation="h",
        marker=dict(color=[T.HIGH if c > 0 else T.LOW for c in d.contribution], cornerradius=4),
        text=[f"{c:+.2f}" for c in d.contribution], textposition="outside", cliponaxis=False,
        textfont=dict(color=T.INK, size=11), hovertemplate="%{y}<br>%{x:+.2f} log-odds<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=T.INK_3, width=1))
    T.style_figure(fig, height=330, legend=False)
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=12, t=14, b=8))
    lo, hi = min(0.0, float(d.contribution.min())), max(0.0, float(d.contribution.max()))
    pad = 0.25 * max(hi - lo, 0.5)                   # room for the outside value labels
    fig.update_xaxes(title_text="← lowers risk · log-odds · raises risk →", title_font=dict(size=11, color=T.INK_2),
                     range=[lo - (pad if lo < 0 else 0.05), hi + (pad if hi > 0 else 0.05)])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=T.INK, size=11.5))
    return fig


def projection_band(res: WhatIfResult, intervals: dict | None, coverage: float = 0.8) -> pd.DataFrame | None:
    """The scenario projection's calibrated band for the next 2 h (fitted twin only)."""
    if intervals is None or res.mode != "fitted":
        return None
    p = res.projection
    hist, fut = p[p.phase == "history"], p[p.phase != "history"]
    start = hist.cgm_scenario.dropna()
    if start.empty or fut.empty:
        return None
    lo, hi = band_for(intervals, coverage, float(start.iloc[-1]))
    n = min(len(fut), len(lo))
    f = fut.iloc[:n]
    return pd.DataFrame({"timestamp": f.timestamp.to_numpy(), "lower": np.clip(f.sim_scenario.to_numpy() + lo[:n], 40, 400),
                         "upper": np.clip(f.sim_scenario.to_numpy() + hi[:n], 40, 400)})


def projection_figure(res: WhatIfResult, thresholds_mgdl: dict[str, float], band: pd.DataFrame | None = None) -> go.Figure:
    p = res.projection
    now = res.baseline.time
    fig = go.Figure()
    if band is not None:
        fig.add_trace(go.Scatter(x=band.timestamp, y=band.upper, mode="lines", line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=band.timestamp, y=band.lower, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(15,118,110,0.16)", name="80% band (twin's typical error)",
                                 customdata=band.upper, hovertemplate="%{y:.0f}-%{customdata:.0f} mg/dL<extra>80% band</extra>"))
    fig.add_vrect(x0=now, x1=p.timestamp.iloc[-1], fillcolor="rgba(15,118,110,0.06)", line_width=0)
    fig.add_annotation(x=now, y=1, yref="paper", text="  next 2 h →", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=11, color=T.BRAND, family=T.FONT))
    fig.add_hline(y=thresholds_mgdl["hyper_mgdl"], line=dict(color=T.HIGH, width=1.2, dash="dash"))
    fig.add_hline(y=thresholds_mgdl["hypo_mgdl"], line=dict(color=T.LOW, width=1.2, dash="dash"))
    fitted = res.mode == "fitted"
    if fitted:
        fig.add_trace(go.Scatter(x=p.timestamp, y=p.cgm_baseline, name="Measured CGM", mode="markers",
                                 marker=dict(color=T.INK_3, size=4, opacity=0.6),
                                 hovertemplate="%{y:.0f} mg/dL<extra>measured CGM</extra>"))
    base_name = "Twin estimate, as logged" if fitted else "As it happened (hidden truth)"
    fig.add_trace(go.Scatter(x=p.timestamp, y=p.sim_baseline, name=base_name, mode="lines",
                             line=dict(color=T.INK_2, width=2.2, dash="dot"),
                             hovertemplate="%{y:.0f} mg/dL<extra>" + base_name + "</extra>"))
    fig.add_trace(go.Scatter(x=p.timestamp, y=p.sim_scenario, name="With scenario", mode="lines",
                             line=dict(color=T.BRAND, width=3), hovertemplate="%{y:.0f} mg/dL<extra>with scenario</extra>"))
    fig.add_vline(x=now, line=dict(color=T.BRAND, width=2))
    T.style_figure(fig, height=360)
    top = float(np.nanmax(p[["sim_baseline", "sim_scenario", "cgm_baseline"]].to_numpy()))
    fig.update_yaxes(title_text="mg/dL", range=[40, max(260, top + 20)])
    return fig


# --------------------------------------------------------------------------- views
def view_overview(ctx: TwinContext, twin: PatientTwin, series: pd.DataFrame, tl: pd.DataFrame) -> None:
    e = twin.profile
    left, right = st.columns([1, 1], gap="large")
    with left:
        html(T.section_html("Profile", "From the synthetic EHR record"))
        meds = []
        if e["on_metformin"]:
            meds.append(f"Metformin {e['metformin_dose_mg']:.0f} mg/day")
        if e["on_sulfonylurea"]:
            meds.append(f"Sulfonylurea (glimepiride-eq.) {e['sulfonylurea_dose_mg']:.0f} mg/day, {e['sulfonylurea_frequency']}")
        if e["on_insulin"]:
            kind = "premixed 30/70, twice daily" if e["insulin_regimen"] == "premix_bid" else "basal, at bedtime"
            meds.append(f"Insulin {e['insulin_total_daily_units']:.0f} U/day ({kind})")
        comorb = [n for k, n in (("hypertension", "Hypertension"), ("dyslipidemia", "Dyslipidaemia"),
                                 ("coronary_artery_disease", "Coronary artery disease")) if e[k]]
        html(T.table_html([
            ("Age / sex", f"{e['age']} years, {'male' if e['sex'] == 'M' else 'female'}"),
            ("BMI", f"{e['bmi']} kg/m² ({e['weight_kg']:.0f} kg)"),
            ("Diabetes duration", f"{e['diabetes_duration_years']} years"),
            ("Medication", "; ".join(meds) or "No glucose-lowering drugs"),
            ("Comorbidities", "; ".join(comorb) or "None recorded"),
            ("Family history of diabetes", "Yes" if e["family_history_diabetes"] else "No"),
            ("Severe hypo, past 12 months", "Yes" if e["prior_severe_hypo_12m"] else "No"),
            ("ICD-10 codes", str(e["diagnoses_icd10"]).replace(";", ", ")),
        ]))
        html(T.section_html("Latest labs"))
        labs = st.columns(4)
        labs[0].metric("HbA1c", f"{e['hba1c']}%")
        labs[1].metric("eGFR", f"{e['egfr']:.0f}", help="mL/min/1.73m²; < 60 suggests reduced kidney function")
        labs[2].metric("LDL", f"{e['ldl_mgdl']:.0f}", help="mg/dL")
        labs[3].metric("TG", f"{e['triglycerides_mgdl']:.0f}", help="Triglycerides, mg/dL")
    with right:
        s = cgm_summary(series)
        html(T.section_html("CGM summary, 14 days", f"Data coverage {s['coverage']:.0%} · consensus targets in tooltips"))
        html(T.tir_bar_html(s["tbr"], s["tir"], s["tar"]))
        st.write("")
        c = st.columns(3)
        c[0].metric("In range", f"{s['tir']:.0%}", help="Time in range 70-180 mg/dL. Consensus target: > 70%")
        c[1].metric("Below 70", f"{s['tbr']:.1%}", help="Time below 70 mg/dL. Consensus target: < 4%")
        c[2].metric("Above 180", f"{s['tar']:.0%}", help="Time above 180 mg/dL. Consensus target: < 25%")
        c = st.columns(3)
        c[0].metric("Mean", f"{s['mean']:.0f}", help="Mean CGM glucose, mg/dL")
        c[1].metric("GMI", f"{s['gmi']:.1f}%", help="Glucose management indicator: CGM-estimated HbA1c")
        c[2].metric("Variability", f"{s['cv']:.0%}", help="Coefficient of variation. Stable: <= 36%")
        days = len(tl) * ctx.cfg["windows"]["stride_minutes"] / (60 * 24)
        html(T.section_html("Alerts over the record", "Raised by the alert policy: one per risk episode, repeats suppressed"))
        c = st.columns(2)
        for col, t, name in ((c[0], "hypo", "Hypo alerts"), (c[1], "hyper", "Hyper alerts")):
            n, above = int(tl[f"{t}_alert"].sum()), int(tl[f"{t}_above"].sum())
            col.metric(name, n, help=f"{n / max(days, 1e-9):.1f} per day over {days:.0f} monitored days. Without the policy, "
                                    f"{above} prediction windows were above the threshold.")


def view_timeline(ctx: TwinContext, twin: PatientTwin, series: pd.DataFrame, tl: pd.DataFrame) -> None:
    c1, c2 = st.columns([2, 3])
    with c1:
        hours = st.select_slider("Hours of history shown", options=[6, 12, 24, 48, 72], value=24)
    with c2:
        if twin.outcome is not None:
            items = [(f"{TARGET_NAME[t]} in next 2 h", "yes" if v else "no") for t, v in twin.outcome.items()]
            html('<div style="margin-top:28px">' + T.chips_html([("What actually happened ·", "synthetic ground truth")] + items)
                 + "</div>")
    st.plotly_chart(timeline_figure(series, tl, twin.time, hours, ctx.thresholds), use_container_width=True, config=T.PLOTLY_CONFIG)
    st.caption("Shaded bands mark high (> 180 mg/dL) and low (< 70 mg/dL) glucose. Filled triangles are alerts raised by "
               "the alert policy; open triangles are times when risk was still at or above the dashed threshold but no "
               "new alert was raised. The dotted trace after NOW is what actually happened; the model had not seen it.")
    html(T.chips_html([("Alert policy ·", "chosen on validation patients, see docs/alert_policy.md")]
                      + [(f"{TARGET_NAME[t]} ·", ctx.policies[t].describe()) for t in ("hypo", "hyper")]))


def view_replay(ctx: TwinContext, twin: PatientTwin, series: pd.DataFrame, tl: pd.DataFrame) -> None:
    html(T.section_html("Live replay",
                        "Plays this patient's record forward as if it were happening now. At each moment the model "
                        "has seen only the data up to the moving line."))
    c1, c2 = st.columns([3, 2])
    with c1:
        hours = st.select_slider("Start this many hours before the prediction time", options=[4, 6, 8, 12], value=6)
    with c2:
        speed = st.radio("Playback speed", ["Normal", "Fast"], horizontal=True)
    start = max(twin.time - pd.Timedelta(hours=hours), series.timestamp.iloc[0])
    end = min(twin.time + pd.Timedelta(hours=2), series.timestamp.iloc[-1])
    with st.spinner("Preparing the replay…"):
        eps = R.episodes(series, tl, start, end, ctx.cfg["thresholds"], alert_reasons(twin.patient_id, start, end, ctx.model_name))
        fig = R.replay_figure(series, tl, start, end, ctx.thresholds, ctx.cfg["thresholds"], eps,
                              frame_ms=350 if speed == "Normal" else 120)
    st.plotly_chart(fig, use_container_width=True, config=T.PLOTLY_CONFIG)
    if eps:
        html(T.section_html("What happens in this replay",
                            "Every alert raised in the window, and any episode (>= 15 min past the threshold) it missed"))
        st.markdown("\n".join(f"- {e.sentence()}" for e in eps))
    else:
        st.caption("No alerts and no glucose episodes in this window. Pick another time in the sidebar, e.g. "
                   "“Highest hypoglycaemia risk”.")
    st.caption("Press Play, or drag the slider. Alerts appear when the alert policy raised them, with the two "
               "drivers that pushed the risk up most. Glucose after the moving line is not shown until it happens.")


def view_drivers(ctx: TwinContext, twin: PatientTwin, tl: pd.DataFrame) -> None:
    html(T.section_html("Why the model raised (or did not raise) each risk",
                        "TreeSHAP contributions in log-odds. They explain the model, not the physiology."))
    cols = st.columns(2, gap="large")
    for col, target in zip(cols, ("hypo", "hyper")):
        with col:
            state, _ = alert_state(tl, twin.anchor_idx, target)
            label, kind = T.ALERT_STATES[state]
            pill = T.pill_html(label, kind, ("▼" if target == "hypo" else "▲") if state != "ok" else "✓")
            html(f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0 8px">'
                 f'<span style="font-family:{T.HEAD_FONT};font-weight:800;font-size:1.25rem;color:{T.INK}">'
                 f'{TARGET_NAME[target]} · {twin.risk[target]:.0%}</span>{pill}</div>')
            d = explain(ctx, twin, target, top_k=7)
            st.plotly_chart(drivers_figure(d), use_container_width=True, config=T.PLOTLY_CONFIG)
            for line in driver_sentences(d.head(3), target):
                st.markdown(f"- {line}")
            g = group_contributions(ctx, twin, target)
            html(T.chips_html([("By data stream ·", "log-odds")] + [(k, f"{v:+.2f}") for k, v in g.items()]))


def _meal_options(ev: pd.DataFrame, past_only: bool) -> dict[str, int]:
    m = ev[ev.event_type == "meal"]
    if past_only:
        m = m[m.hours_from_now <= 0]
    return {f"{r.detail.replace('_', ' ')} ({r.value:.0f} g) at {r.hours_from_now:+.1f} h": int(r.step) for r in m.itertuples()}


def preset_scenarios(name: str, ev: pd.DataFrame, now: pd.Timestamp, on_insulin: bool) -> tuple[list[Scenario], str]:
    """The four demo presets. Returns (scenarios, explanation)."""
    meals = ev[(ev.event_type == "meal") & ev.detail.isin(MAIN_MEALS)]
    if name == "Late dinner (+2 h)":
        clock = [(now + pd.Timedelta(hours=h)).hour for h in meals.hours_from_now]
        dinner = meals[[c >= 18 for c in clock]]
        if dinner.empty:
            return [], ("No dinner within 6 h before or 2 h after this prediction time. Pick an evening time "
                        "(sidebar → Choose a time, e.g. 19:00-22:00) to try a late dinner.")
        d = dinner.iloc[-1]
        return [DelayMeal(int(d.step), 2.0)], (f"Moves dinner ({d.detail.replace('_', ' ')}, {d.hours_from_now:+.1f} h) "
                                               "two hours later; medication at the usual time.")
    if name == "Poor sleep night":
        return [PoorSleep()], "Last night becomes short and fragmented; insulin resistance rises for the day."
    if name == "Skipped meal":
        past = meals[meals.hours_from_now <= 0]
        if past.empty:
            return [], "No main meal in the last 6 h to skip."
        last = past.iloc[-1]
        return [SkipMeal(int(last.step))], f"Skips the {last.detail.replace('_', ' ')} at {last.hours_from_now:+.1f} h; medication still taken."
    if name == "Extra insulin dose":
        note = "An extra 8 U of premixed insulin 1 h ago (e.g. a double injection)."
        if not on_insulin:
            note += " This patient is not on insulin, so it shows the effect of an accidental dose."
        return [ExtraDose("premix", 8.0, -1.0)], note
    return [], ""


def fit_chips(sim: PatientSimulator) -> list[tuple[str, str]]:
    fit = sim.fit
    names = str(fit.info.get("fitted_params", ",".join(FIT_PARAMS))).split(",")
    items = [(f"{PARAM_LABEL[n]} ·", f"{fit.params.iloc[0][n]:.3g} (EHR default {fit.default.iloc[0][n]:.3g})")
             for n in names]
    return [("Fit error, first week ·", f"{float(fit.info['fit_mae']):.1f} mg/dL MAE")] + items


def view_whatif(ctx: TwinContext, twin: PatientTwin) -> None:
    html(T.section_html("Ask the twin “what if?”",
                        "Two separate answers: what the risk model predicts from the data it would see, and what the "
                        "simulated physiology does over the next 2 hours."))
    if not st.session_state.get("whatif_on"):
        # The twin is built and re-simulated on demand, so the first page load does not pay for it.
        st.markdown("Try a **late dinner**, a **poor night's sleep**, a **skipped meal** or an **extra insulin dose**, "
                    "or build your own scenario. The twin is fitted to this patient's own first week of data.")
        if st.button("Start the what-if twin", type="primary", key="start_whatif"):
            st.session_state["whatif_on"] = True
            st.rerun()
        return
    label = st.radio("Twin physiology", list(TWIN_MODES), horizontal=True,
                     help="Personalised: parameters fitted to this patient's first week of CGM and their meal / dose "
                          "log, as a real twin would have to. Oracle: replays the simulator's hidden truth, which is "
                          "only possible because the data is synthetic.")
    mode = TWIN_MODES[label]
    sim = get_simulator(twin.patient_id, mode)
    if mode == "oracle":
        st.warning("Oracle mode uses the simulator's **hidden** parameters, schedule and noise. No real system could do "
                   "this; it is shown only to compare against the personalised twin.")
    ev = sim.events(twin.anchor_idx)
    preset = st.radio("Scenario", ["Late dinner (+2 h)", "Poor sleep night", "Skipped meal", "Extra insulin dose", "Custom"],
                      horizontal=True)
    if preset != "Custom":
        scenarios, note = preset_scenarios(preset, ev, twin.time, bool(twin.profile["on_insulin"]))
        st.caption(note)
    else:
        scenarios = []
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("**Meals**")
            if st.checkbox("Add a meal"):
                meal = st.selectbox("Meal", list(MEAL_CATALOGUE), format_func=lambda m: m.replace("_", " "))
                h = st.slider("When (hours from now)", -6.0, 2.0, -1.0, 0.5, key="meal_h")
                scenarios.append(AddMeal(meal, h))
            past = _meal_options(ev, past_only=True)
            if past and st.checkbox("Skip a recent meal"):
                scenarios.append(SkipMeal(past[st.selectbox("Meal to skip", list(past))]))
            allm = _meal_options(ev, past_only=False)
            if allm and st.checkbox("Delay a meal"):
                step = allm[st.selectbox("Meal to delay", list(allm))]
                scenarios.append(DelayMeal(step, st.slider("Delay (hours)", 0.5, 4.0, 2.0, 0.5)))
        with c2:
            st.markdown("**Sleep and activity**")
            if st.checkbox("Poor sleep last night"):
                scenarios.append(PoorSleep())
            if st.checkbox("Add a brisk walk"):
                h = st.slider("Start (hours from now)", -6.0, 1.5, -1.0, 0.5, key="walk_h")
                scenarios.append(AddActivity(h, st.slider("Duration (min)", 15, 120, 45, 15)))
        with c3:
            st.markdown("**Medication**")
            if st.checkbox("Extra dose"):
                kind = st.selectbox("Type", ["premix", "rapid", "basal", "sulfonylurea"])
                amt = st.number_input("Amount (U, or mg for sulfonylurea)", 0.5, 30.0, 8.0 if kind != "sulfonylurea" else 2.0, 0.5)
                h = st.slider("When (hours from now)", -6.0, 2.0, -1.0, 0.5, key="dose_h")
                scenarios.append(ExtraDose(kind, float(amt), h))
            on_drug = bool(twin.profile["on_insulin"] or twin.profile["on_sulfonylurea"])
            if on_drug and st.checkbox("Change prescribed dose"):
                scenarios.append(ScaleMedication(st.slider("Dose factor", 0.25, 2.0, 1.5, 0.25)))
            elif not on_drug:
                st.caption("Not on insulin or a sulfonylurea: no dose to change.")

    if not scenarios:
        st.info("Choose at least one change." if preset == "Custom" else "This preset does not apply here (see above).")
        return
    res = cached_whatif(ctx, sim, twin.patient_id, twin.anchor_idx, mode, scenarios)

    html(T.chips_html([("Scenario ·", "; ".join(res.descriptions))]))
    st.write("")
    cols = st.columns(2, gap="large")
    for col, t in zip(cols, ("hypo", "hyper")):
        with col:
            html(T.compare_card_html(t, TARGET_NAME[t], res.baseline.risk[t], res.scenario.risk[t], res.scenario.alert[t]))
    if res.unseen_by_model:
        st.warning("The risk model **cannot see** changes after *now* (" + "; ".join(res.unseen_by_model) + "). "
                   "Its risk only reflects past data; the simulator projection below does include them.")
    if mode == "fitted":
        html(T.section_html("Simulated glucose · personalised twin",
                            "Parameters fitted to this patient's first week; pulled toward the measured CGM up to now, "
                            "then free-running. Estimate as logged vs with the scenario."))
    else:
        html(T.section_html("Simulated glucose · oracle replay",
                            "The simulator's hidden truth: as it happened vs with the scenario"))
    intervals = get_intervals()
    band = projection_band(res, intervals)
    st.plotly_chart(projection_figure(res, ctx.cfg["thresholds"], band), use_container_width=True, config=T.PLOTLY_CONFIG)
    o = res.simulated_outcome
    chips = [
        ("Lowest, next 2 h", f"{o['baseline']['sim_min']:.0f} → {o['scenario']['sim_min']:.0f} mg/dL"),
        ("Highest, next 2 h", f"{o['baseline']['sim_max']:.0f} → {o['scenario']['sim_max']:.0f} mg/dL"),
        ("CGM low", "yes" if o["scenario"]["hypo"] else "no"),
        ("CGM high", "yes" if o["scenario"]["hyper"] else "no"),
    ]
    if band is not None:
        chips.insert(1, ("Lowest, 80% band", f"{band.lower.min():.0f}-{band.upper.min():.0f} mg/dL"))
    html(T.chips_html(chips))
    if band is not None:
        cov = intervals.get("test_coverage", {}).get("0.8")
        st.caption("Shaded: the 80% band, i.e. how far this twin's 2-hour projection usually misses, calibrated on "
                   "validation patients' held-out days"
                   + (f"; on test patients the real CGM fell inside it {cov:.0%} of the time" if cov else "")
                   + " (docs/twin_intervals.md). It covers the projection as logged; how the twin responds to the "
                     "scenario's change adds error the band does not include.")
    if mode == "fitted":
        html(T.chips_html(fit_chips(sim)))
        st.caption("Personalised twin: the simulator's equations with four parameters fitted per patient (see "
                   "docs/twin_personalisation.md). Inputs are the EHR, the first week of CGM, the logged meals and "
                   "doses, and the wearable; no hidden simulator information is used. The equations are still the "
                   "synthetic model's own, so this is a best case, not clinical evidence.")
    else:
        st.caption("Oracle replay of the synthetic simulator, including its hidden parameters and noise. In real use "
                   "there would be no ground-truth physiology to replay; use the personalised twin instead.")


# --------------------------------------------------------------------------- page
def pick_time(ctx: TwinContext, pid: str, tl: pd.DataFrame) -> int:
    """Sidebar control for the prediction time; returns the anchor index."""
    mode = st.sidebar.radio("Prediction time (now)", ["Highest hypoglycaemia risk", "Highest hyperglycaemia risk", "Choose a time"])
    onset = tl[~tl.in_event_hypo & ~tl.in_event_hyper]
    pool = onset if len(onset) else tl
    if mode == "Highest hypoglycaemia risk":
        return int(pool.loc[pool.hypo_risk.idxmax(), "anchor_idx"])
    if mode == "Highest hyperglycaemia risk":
        return int(pool.loc[pool.hyper_risk.idxmax(), "anchor_idx"])
    days = sorted(tl.timestamp.dt.date.unique())
    day = st.sidebar.selectbox("Day", days, index=len(days) // 2, format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b"))
    opts = tl[tl.timestamp.dt.date == day]
    ts = st.sidebar.select_slider("Time", options=list(opts.timestamp), value=opts.timestamp.iloc[len(opts) // 2],
                                  format_func=lambda t: pd.Timestamp(t).strftime("%H:%M"))
    return int(opts.loc[opts.timestamp == ts, "anchor_idx"].iloc[0])


def main() -> None:
    st.set_page_config(page_title="Glycaemic Digital Twin", page_icon=str(ROOT / "src" / "app" / "assets" / "favicon.png"),
                       layout="wide")
    html(T.CSS)
    html(T.banner_html())
    try:
        ctx = get_context()
    except FileNotFoundError as exc:
        st.error(f"Pipeline outputs not found ({exc.filename}). Run `scripts\\run_pipeline.cmd` (Windows) or "
                 "`bash scripts/run_pipeline.sh` first.")
        st.stop()

    with st.sidebar:
        html(T.brand_html())
    models = {k: (v or ctx.cfg["twin"]["model"]) for k, v in RISK_MODELS.items()
              if v is None or TwinContext.model_available(ctx.cfg, v)}
    choice = st.sidebar.radio("Risk model", list(models), help=(
        "Synthetic-trained: matches the synthetic patients shown here. Real-patient-trained: trained and "
        "alert-tested on 100 real patients; on real data the EHR and wearables did not improve the forecast, "
        "so it uses CGM only (docs/real_model.md)."))
    ctx = get_model_context(models[choice])
    pts = patient_table()
    only_hypo = st.sidebar.checkbox("Only patients with hypoglycaemia events", value=True)
    show = pts[pts.hypo_events > 0] if only_hypo and (pts.hypo_events > 0).any() else pts
    labels = {r.patient_id: f"{r.patient_id} · {r.regimen} · HbA1c {r.hba1c}%" for r in show.itertuples()}
    pid = st.sidebar.selectbox("Virtual patient (test set)", list(labels), format_func=labels.get)
    tl = get_timeline(pid, ctx.model_name)
    anchor = pick_time(ctx, pid, tl)
    twin = ctx.build_twin(pid, anchor)
    series = get_series(pid)
    st.sidebar.markdown("---")
    info = real_model_info(ctx)
    if info is None:
        st.sidebar.caption(f"Model: LightGBM on CGM + wearables + EHR, trained on synthetic patients. Test patients only "
                           f"({len(pts)}); none were used for training, model selection, alert thresholds or the alert policy.")
    else:
        a = info.get("real_patient_alerts", {}).get("hypo")
        st.sidebar.caption(f"Model: LightGBM on CGM + time of day, trained on {info['n_patients']} real patients "
                           "(ShanghaiT2DM, CC BY 4.0); thresholds set on real patients. It has never seen these synthetic patients."
                           + (f" On real patients its alerts warned of {a['caught_30']:.0%} of lows at least 30 min ahead "
                              f"({a['alerts_per_day']:.2f} alerts a day, {a['ppv']:.0%} followed by a real low)." if a else ""))
    if str(ctx.cfg["paths"]["raw_dir"]).startswith("demo_bundle"):
        st.sidebar.caption("Running from the committed demo bundle (precomputed synthetic data and model).")

    e = twin.profile
    html(T.hero_html(pid, [f"{e['age']} y · {'Male' if e['sex'] == 'M' else 'Female'}", regimen_label(e["regimen"]),
                           f"HbA1c {e['hba1c']}%", f"eGFR {e['egfr']:.0f}"],
                     fmt_time(twin.time), "Uses the previous 6 h of CGM only" if ctx.feature_set == "cgm"
                     else "Uses the previous 6 h of data + the EHR profile"))
    c1, c2 = st.columns(2, gap="large")
    for col, t in zip((c1, c2), ("hypo", "hyper")):
        with col:
            state, status = alert_state(tl, twin.anchor_idx, t)
            html(T.risk_card_html(t, TARGET_NAME[t], TARGET_RULE[t], twin.risk[t], ctx.thresholds[t], state, status,
                                  far_label(ctx, t)))

    tabs = st.tabs(["Live replay", "Glucose timeline", "Alert drivers", "What-if", "Patient overview"])
    with tabs[0]:
        view_replay(ctx, twin, series, tl)
    with tabs[1]:
        view_timeline(ctx, twin, series, tl)
    with tabs[2]:
        view_drivers(ctx, twin, tl)
    with tabs[3]:
        view_whatif(ctx, twin)
    with tabs[4]:
        view_overview(ctx, twin, series, tl)


if __name__ == "__main__":
    main()
