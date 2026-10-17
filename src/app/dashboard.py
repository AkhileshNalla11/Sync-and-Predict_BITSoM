"""Clinician dashboard for the glycaemic digital twin (Streamlit).

    streamlit run src/app/dashboard.py        # or: scripts\\demo.cmd / bash scripts/demo.sh

Views: glucose timeline with both risks and alerts, alert drivers, what-if panel, patient overview.
SYNTHETIC DATA. Research prototype, NOT FOR CLINICAL USE. Styling lives in src/app/theme.py.
"""
from __future__ import annotations

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

from src.app import theme as T                      # noqa: E402
from src.config import load_config                  # noqa: E402
from src.twin.explain import driver_sentences, explain, group_contributions   # noqa: E402
from src.twin.state import PatientTwin, TwinContext  # noqa: E402
from src.twin.whatif import (MAIN_MEALS, MEAL_CATALOGUE, AddActivity, AddMeal, DelayMeal, ExtraDose,  # noqa: E402
                             PatientSimulator, PoorSleep, Scenario, ScaleMedication, SkipMeal, WhatIfResult,
                             run_whatif)

TARGET_NAME = {"hyper": "Hyperglycaemia", "hypo": "Hypoglycaemia"}
TARGET_RULE = {"hyper": "Any glucose > 180 mg/dL in the next 2 h", "hypo": "Any glucose < 70 mg/dL in the next 2 h"}
HIGH_BAND, LOW_BAND = "rgba(235,104,52,0.09)", "rgba(42,120,214,0.10)"


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


@st.cache_resource(show_spinner="Loading the twin (data, model, thresholds)...")
def get_context() -> TwinContext:
    return TwinContext.load(runtime_config())


@st.cache_resource(show_spinner="Replaying this patient's simulation...", max_entries=8)
def get_simulator(patient_id: str) -> PatientSimulator:
    return PatientSimulator(get_context().cfg, patient_id)


@st.cache_data(show_spinner=False, max_entries=32)
def get_timeline(patient_id: str) -> pd.DataFrame:
    return get_context().risk_timeline(patient_id)


@st.cache_data(show_spinner=False, max_entries=32)
def get_series(patient_id: str) -> pd.DataFrame:
    return get_context().series_frame(patient_id)


@st.cache_data(show_spinner=False)
def patient_table() -> pd.DataFrame:
    ctx = get_context()
    rows = []
    for pid in ctx.patients("test"):
        tl = get_timeline(pid)
        e = ctx.ehr.loc[pid]
        rows.append({"patient_id": pid, "regimen": e.regimen, "hba1c": e.hba1c, "age": e.age, "sex": e.sex,
                     "hypo_alerts": int(tl.hypo_alert.sum()), "hyper_alerts": int(tl.hyper_alert.sum()),
                     "hypo_events": int(tl.hypo.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- helpers
def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def fmt_time(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%a %d %b · %H:%M")


def regimen_label(regimen: str) -> str:
    return regimen.replace("+", " + ").replace("sulfonylurea", "sulfonylurea").capitalize()


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
        color = T.TARGET_COLOR[t]
        fig.add_trace(go.Scatter(x=a.timestamp, y=np.full(len(a), y), mode="markers", name=f"{TARGET_NAME[t]} alert",
                                 marker=dict(symbol=sym, size=11, color=color, line=dict(color=T.CARD, width=1.5)),
                                 hovertemplate=f"{TARGET_NAME[t]} alert<extra></extra>"), 1, 1)
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
    T.style_figure(fig, height=660)
    fig.update_layout(bargap=0.1)
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


def projection_figure(res: WhatIfResult, thresholds_mgdl: dict[str, float]) -> go.Figure:
    p = res.projection
    now = res.baseline.time
    fig = go.Figure()
    fig.add_vrect(x0=now, x1=p.timestamp.iloc[-1], fillcolor="rgba(15,118,110,0.06)", line_width=0)
    fig.add_annotation(x=now, y=1, yref="paper", text="  next 2 h →", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=11, color=T.BRAND, family=T.FONT))
    fig.add_hline(y=thresholds_mgdl["hyper_mgdl"], line=dict(color=T.HIGH, width=1.2, dash="dash"))
    fig.add_hline(y=thresholds_mgdl["hypo_mgdl"], line=dict(color=T.LOW, width=1.2, dash="dash"))
    fig.add_trace(go.Scatter(x=p.timestamp, y=p.true_baseline, name="As it happened", mode="lines",
                             line=dict(color=T.INK_3, width=2.2, dash="dot"),
                             hovertemplate="%{y:.0f} mg/dL<extra>as it happened</extra>"))
    fig.add_trace(go.Scatter(x=p.timestamp, y=p.true_scenario, name="With scenario", mode="lines",
                             line=dict(color=T.BRAND, width=3), hovertemplate="%{y:.0f} mg/dL<extra>with scenario</extra>"))
    fig.add_vline(x=now, line=dict(color=T.BRAND, width=2))
    T.style_figure(fig, height=360)
    top = float(np.nanmax(p[["true_baseline", "true_scenario"]].to_numpy()))
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
        html(T.section_html("Alerts over the record", "Alerting 30-minute prediction windows, not separate episodes"))
        c = st.columns(2)
        c[0].metric("Hypo alerts", int(tl.hypo_alert.sum()))
        c[1].metric("Hyper alerts", int(tl.hyper_alert.sum()))


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
    st.caption("Shaded bands mark high (> 180 mg/dL) and low (< 70 mg/dL) glucose. Triangles are alerts (risk at or above "
               "the dashed threshold). The dotted trace after NOW is what actually happened; the model had not seen it.")


def view_drivers(ctx: TwinContext, twin: PatientTwin) -> None:
    html(T.section_html("Why the model raised (or did not raise) each risk",
                        "TreeSHAP contributions in log-odds. They explain the model, not the physiology."))
    cols = st.columns(2, gap="large")
    for col, target in zip(cols, ("hypo", "hyper")):
        with col:
            pill = (T.pill_html("ALERT", "alert", "▼" if target == "hypo" else "▲") if twin.alert[target]
                    else T.pill_html("Below threshold", "ok", "✓"))
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


def view_whatif(ctx: TwinContext, twin: PatientTwin) -> None:
    sim = get_simulator(twin.patient_id)
    ev = sim.events(twin.anchor_idx)
    html(T.section_html("Ask the twin “what if?”",
                        "Two separate answers: what the risk model predicts from the data it would see, and what the "
                        "simulated physiology does over the next 2 hours."))
    mode = st.radio("Scenario", ["Late dinner (+2 h)", "Poor sleep night", "Skipped meal", "Extra insulin dose", "Custom"],
                    horizontal=True)
    if mode != "Custom":
        scenarios, note = preset_scenarios(mode, ev, twin.time, bool(twin.profile["on_insulin"]))
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
        st.info("Choose at least one change." if mode == "Custom" else "This preset does not apply here (see above).")
        return
    with st.spinner("Re-simulating and re-predicting..."):
        res = run_whatif(ctx, sim, twin.anchor_idx, scenarios)

    html(T.chips_html([("Scenario ·", "; ".join(res.descriptions))]))
    st.write("")
    cols = st.columns(2, gap="large")
    for col, t in zip(cols, ("hypo", "hyper")):
        with col:
            html(T.compare_card_html(t, TARGET_NAME[t], res.baseline.risk[t], res.scenario.risk[t], res.scenario.alert[t]))
    if res.unseen_by_model:
        st.warning("The risk model **cannot see** changes after *now* (" + "; ".join(res.unseen_by_model) + "). "
                   "Its risk only reflects past data; the simulator projection below does include them.")
    html(T.section_html("Simulated glucose", "Twin physiology from the synthetic simulator: as it happened vs with the scenario"))
    st.plotly_chart(projection_figure(res, ctx.cfg["thresholds"]), use_container_width=True, config=T.PLOTLY_CONFIG)
    o = res.simulated_outcome
    html(T.chips_html([
        ("Lowest, next 2 h", f"{o['baseline']['true_min']:.0f} → {o['scenario']['true_min']:.0f} mg/dL"),
        ("Highest, next 2 h", f"{o['baseline']['true_max']:.0f} → {o['scenario']['true_max']:.0f} mg/dL"),
        ("CGM low", "yes" if o["scenario"]["hypo"] else "no"),
        ("CGM high", "yes" if o["scenario"]["hyper"] else "no"),
    ]))
    st.caption("The simulator is the synthetic model that generated this data, so the projection is only as good as "
               "that model. In real use there would be no ground-truth physiology to replay.")


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
    st.set_page_config(page_title="Glycaemic Digital Twin", page_icon="🩸", layout="wide")
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
    pts = patient_table()
    only_hypo = st.sidebar.checkbox("Only patients with hypoglycaemia events", value=True)
    show = pts[pts.hypo_events > 0] if only_hypo and (pts.hypo_events > 0).any() else pts
    labels = {r.patient_id: f"{r.patient_id} · {r.regimen} · HbA1c {r.hba1c}%" for r in show.itertuples()}
    pid = st.sidebar.selectbox("Virtual patient (test set)", list(labels), format_func=labels.get)
    tl = get_timeline(pid)
    anchor = pick_time(ctx, pid, tl)
    twin = ctx.build_twin(pid, anchor)
    series = get_series(pid)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model: LightGBM on CGM + wearables + EHR. Test patients only ({len(pts)}); none were used "
                       "for training, model selection or alert thresholds.")
    if str(ctx.cfg["paths"]["raw_dir"]).startswith("demo_bundle"):
        st.sidebar.caption("Running from the committed demo bundle (precomputed synthetic data and model).")

    e = twin.profile
    html(T.hero_html(pid, [f"{e['age']} y · {'Male' if e['sex'] == 'M' else 'Female'}", regimen_label(e["regimen"]),
                           f"HbA1c {e['hba1c']}%", f"eGFR {e['egfr']:.0f}"],
                     fmt_time(twin.time), "Uses the previous 6 h of data + the EHR profile"))
    c1, c2 = st.columns(2, gap="large")
    for col, t in zip((c1, c2), ("hypo", "hyper")):
        with col:
            html(T.risk_card_html(t, TARGET_NAME[t], TARGET_RULE[t], twin.risk[t], ctx.thresholds[t], twin.alert[t]))

    tabs = st.tabs(["Glucose timeline", "Alert drivers", "What-if", "Patient overview"])
    with tabs[0]:
        view_timeline(ctx, twin, series, tl)
    with tabs[1]:
        view_drivers(ctx, twin)
    with tabs[2]:
        view_whatif(ctx, twin)
    with tabs[3]:
        view_overview(ctx, twin, series, tl)


if __name__ == "__main__":
    main()
