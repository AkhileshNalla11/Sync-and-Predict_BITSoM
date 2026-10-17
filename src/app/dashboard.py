"""Clinician dashboard for the glycaemic digital twin (Streamlit).

    streamlit run src/app/dashboard.py        # or: bash scripts/demo.sh

Views: patient overview, glucose timeline with both risks and alerts, alert drivers, what-if panel.
SYNTHETIC DATA. Research prototype, NOT FOR CLINICAL USE.
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

from src.config import load_config                  # noqa: E402
from src.twin.explain import driver_sentences, explain, group_contributions   # noqa: E402
from src.twin.state import PatientTwin, TwinContext  # noqa: E402
from src.twin.whatif import (MAIN_MEALS, MEAL_CATALOGUE, AddActivity, AddMeal, DelayMeal, ExtraDose,  # noqa: E402
                             PatientSimulator, PoorSleep, Scenario, ScaleMedication, SkipMeal, WhatIfResult,
                             run_whatif)

# Palette (validated with the dataviz validator on the light surface). Warm = high, cool = low.
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
HIGH, LOW, NEUTRAL = "#eb6834", "#2a78d6", "#52514e"
HIGH_BAND, LOW_BAND = "rgba(235,104,52,0.10)", "rgba(42,120,214,0.12)"
TARGET_NAME = {"hyper": "Hyperglycaemia", "hypo": "Hypoglycaemia"}
TARGET_RULE = {"hyper": "glucose > 180 mg/dL in the next 2 h", "hypo": "glucose < 70 mg/dL in the next 2 h"}
TARGET_COLOR = {"hyper": HIGH, "hypo": LOW}

BANNER = ("⚠️ **SYNTHETIC DATA. Research prototype, not for clinical use.** Every patient, record and sensor "
          "trace here is simulated, and the link between EHR and wearable data is synthetic by design. "
          "Risk scores have not been clinically validated.")


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


# --------------------------------------------------------------------------- small UI pieces
def fmt_time(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%a %d %b, %H:%M")


def risk_card(twin: PatientTwin, target: str, threshold: float) -> None:
    """Risk with an icon + text label (never color alone)."""
    p = twin.risk[target]
    alert = twin.alert[target]
    icon, status = ("🔺", "ALERT") if alert and target == "hyper" else ("🔻", "ALERT") if alert else ("✓", "below alert threshold")
    with st.container(border=True):
        st.markdown(f"**{TARGET_NAME[target]} risk**  \n<span style='color:{INK_2};font-size:0.85em'>"
                    f"{TARGET_RULE[target]}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size:2.1em;font-weight:600;color:{INK}'>{p:.0%}</span>"
                    f"&nbsp;&nbsp;<span style='color:{TARGET_COLOR[target] if alert else INK_2};font-weight:600'>"
                    f"{icon} {status}</span>", unsafe_allow_html=True)
        st.caption(f"Alert threshold {threshold:.0%} (5% false-alert rate on validation patients)")


def cgm_summary(series: pd.DataFrame) -> dict[str, float]:
    g = series.cgm_mgdl.dropna()
    return {"mean": g.mean(), "gmi": 3.31 + 0.02392 * g.mean(), "tir": ((g >= 70) & (g <= 180)).mean(),
            "tar": (g > 180).mean(), "tbr": (g < 70).mean(), "cv": g.std() / g.mean(),
            "coverage": series.cgm_mgdl.notna().mean()}


def _style_axes(fig: go.Figure) -> None:
    fig.update_layout(plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK_2, size=12),
                      margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor=INK_3, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=INK_3, zeroline=False)


# --------------------------------------------------------------------------- figures
def timeline_figure(series: pd.DataFrame, tl: pd.DataFrame, now: pd.Timestamp, hours_back: int,
                    thresholds: dict[str, float]) -> go.Figure:
    start, end = now - pd.Timedelta(hours=hours_back), now + pd.Timedelta(hours=2)
    s = series[(series.timestamp >= start) & (series.timestamp <= end)]
    past, future = s[s.timestamp <= now], s[s.timestamp >= now]
    r = tl[(tl.timestamp >= start) & (tl.timestamp <= now)]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.3, 0.15],
                        subplot_titles=("Glucose (CGM)", "Predicted risk for the next 2 h", "Activity and sleep"))
    for y0, y1, color in ((180, 400, HIGH_BAND), (40, 70, LOW_BAND)):
        fig.add_shape(type="rect", xref="x", yref="y", x0=start, x1=end, y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")
    fig.add_annotation(x=start, y=190, xref="x", yref="y", text="high (> 180)", showarrow=False, xanchor="left",
                       font=dict(size=10, color=INK_2))
    fig.add_annotation(x=start, y=60, xref="x", yref="y", text="low (< 70)", showarrow=False, xanchor="left",
                       font=dict(size=10, color=INK_2))
    fig.add_trace(go.Scatter(x=past.timestamp, y=past.cgm_mgdl, mode="lines", name="CGM (seen by model)",
                             line=dict(color=INK, width=2), hovertemplate="%{y:.0f} mg/dL<extra>CGM</extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=future.timestamp, y=future.cgm_mgdl, mode="lines", name="Actual next 2 h (not seen by model)",
                             line=dict(color=INK_3, width=2, dash="dot"), hovertemplate="%{y:.0f} mg/dL<extra>actual</extra>"), 1, 1)

    for t, sym, y in (("hyper", "triangle-up", 385), ("hypo", "triangle-down", 48)):
        a = r[r[f"{t}_alert"]]
        fig.add_trace(go.Scatter(x=a.timestamp, y=np.full(len(a), y), mode="markers", name=f"{TARGET_NAME[t]} alert",
                                 marker=dict(symbol=sym, size=10, color=TARGET_COLOR[t], line=dict(color=SURFACE, width=1)),
                                 hovertemplate=f"{TARGET_NAME[t]} alert<extra></extra>"), 1, 1)
        fig.add_trace(go.Scatter(x=r.timestamp, y=r[f"{t}_risk"], mode="lines+markers", name=f"{TARGET_NAME[t]} risk",
                                 line=dict(color=TARGET_COLOR[t], width=2), marker=dict(size=5),
                                 hovertemplate="%{y:.0%}<extra>" + TARGET_NAME[t] + " risk</extra>",
                                 showlegend=False), 2, 1)
        fig.add_hline(y=thresholds[t], line=dict(color=TARGET_COLOR[t], width=1, dash="dash"), row=2, col=1)

    fig.add_trace(go.Bar(x=past.timestamp, y=past.steps, name="Steps / 5 min", marker_color=INK_3,
                         hovertemplate="%{y:.0f} steps<extra></extra>", showlegend=False), 3, 1)
    asleep = past[past.sleep_stage.isin(["light", "deep", "rem"])]
    fig.add_trace(go.Scatter(x=asleep.timestamp, y=np.full(len(asleep), 520), mode="markers", name="Asleep",
                             marker=dict(symbol="square", size=5, color=LOW, opacity=0.5),
                             hovertemplate="asleep (%{text})<extra></extra>", text=asleep.sleep_stage, showlegend=False), 3, 1)

    for row in (1, 2, 3):
        fig.add_vline(x=now, line=dict(color=INK_2, width=1.5), row=row, col=1)
    fig.add_annotation(x=now, y=1, yref="y domain", text="now", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color=INK))
    fig.update_yaxes(range=[40, 400], title_text="mg/dL", row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", row=2, col=1)
    fig.update_yaxes(range=[0, 560], title_text="steps", row=3, col=1)
    fig.update_xaxes(range=[start, end])
    _style_axes(fig)
    fig.update_layout(height=640)
    for a in fig.layout.annotations[:3]:
        a.update(font=dict(size=12, color=INK), x=0, xanchor="left")
    return fig


def drivers_figure(drivers: pd.DataFrame, target: str) -> go.Figure:
    d = drivers.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d.contribution, y=[f"{r.label}: {r.value}" for r in d.itertuples()], orientation="h",
        marker_color=[HIGH if c > 0 else LOW for c in d.contribution],
        text=[f"{c:+.2f}" for c in d.contribution], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:+.2f} log-odds<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK_3, width=1))
    _style_axes(fig)
    fig.update_layout(height=320, hovermode="closest", showlegend=False,
                      title=dict(text=f"Top drivers of {TARGET_NAME[target].lower()} risk", font=dict(size=13, color=INK)))
    lo, hi = min(0.0, float(d.contribution.min())), max(0.0, float(d.contribution.max()))
    pad = 0.25 * max(hi - lo, 0.5)                   # room for the outside value labels
    fig.update_xaxes(title_text="← lowers risk · log-odds · raises risk →",
                     range=[lo - (pad if lo < 0 else 0.05), hi + (pad if hi > 0 else 0.05)])
    return fig


def projection_figure(res: WhatIfResult, thresholds_mgdl: dict[str, float]) -> go.Figure:
    p = res.projection
    now = res.baseline.time
    fig = go.Figure()
    fig.add_vrect(x0=now, x1=p.timestamp.iloc[-1], fillcolor="rgba(82,81,78,0.06)", line_width=0)
    fig.add_annotation(x=now, y=1, yref="paper", text=" next 2 h →", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=11, color=INK_2))
    fig.add_hline(y=thresholds_mgdl["hyper_mgdl"], line=dict(color=HIGH, width=1, dash="dash"))
    fig.add_hline(y=thresholds_mgdl["hypo_mgdl"], line=dict(color=LOW, width=1, dash="dash"))
    fig.add_trace(go.Scatter(x=p.timestamp, y=p.true_baseline, name="As it happened", mode="lines",
                             line=dict(color=INK_3, width=2), hovertemplate="%{y:.0f} mg/dL<extra>as it happened</extra>"))
    fig.add_trace(go.Scatter(x=p.timestamp, y=p.true_scenario, name="With scenario", mode="lines",
                             line=dict(color=INK, width=2.5), hovertemplate="%{y:.0f} mg/dL<extra>with scenario</extra>"))
    fig.add_vline(x=now, line=dict(color=INK_2, width=1.5))
    _style_axes(fig)
    fig.update_layout(height=360)
    fig.update_yaxes(title_text="mg/dL", range=[40, max(260, float(np.nanmax(p[["true_baseline", "true_scenario"]].to_numpy())) + 20)])
    return fig


# --------------------------------------------------------------------------- views
def view_overview(ctx: TwinContext, twin: PatientTwin, series: pd.DataFrame, tl: pd.DataFrame) -> None:
    e = twin.profile
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Profile")
        meds = []
        if e["on_metformin"]:
            meds.append(f"Metformin {e['metformin_dose_mg']:.0f} mg/day")
        if e["on_sulfonylurea"]:
            meds.append(f"Sulfonylurea (glimepiride-equivalent) {e['sulfonylurea_dose_mg']:.0f} mg/day, {e['sulfonylurea_frequency']}")
        if e["on_insulin"]:
            kind = "premixed 30/70, twice daily" if e["insulin_regimen"] == "premix_bid" else "basal, at bedtime"
            meds.append(f"Insulin {e['insulin_total_daily_units']:.0f} U/day ({kind})")
        comorb = [n for k, n in (("hypertension", "Hypertension"), ("dyslipidemia", "Dyslipidaemia"),
                                 ("coronary_artery_disease", "Coronary artery disease")) if e[k]]
        values = [
            f"{e['age']} y, {'male' if e['sex'] == 'M' else 'female'}",
            f"{e['bmi']} kg/m² ({e['weight_kg']:.0f} kg)",
            f"{e['diabetes_duration_years']} years",
            "; ".join(meds) or "No glucose-lowering drugs",
            "; ".join(comorb) or "None recorded",
            "Yes" if e["family_history_diabetes"] else "No",
            "Yes" if e["prior_severe_hypo_12m"] else "No",
            str(e["diagnoses_icd10"]).replace(";", ", "),
        ]
        fields = ["Age / sex", "BMI", "Diabetes duration", "Medication", "Comorbidities",
                  "Family history of diabetes", "Severe hypo, past 12 months", "ICD-10 codes"]
        rows = [f"| {f} | {v} |" for f, v in zip(fields, values)]
        st.markdown("\n".join(["| Field | Value |", "|---|---|", *rows]))
        st.subheader("Latest labs")
        labs = st.columns(4)
        labs[0].metric("HbA1c", f"{e['hba1c']}%")
        labs[1].metric("eGFR", f"{e['egfr']:.0f}", help="mL/min/1.73m²; < 60 suggests reduced kidney function")
        labs[2].metric("LDL (mg/dL)", f"{e['ldl_mgdl']:.0f}")
        labs[3].metric("TG (mg/dL)", f"{e['triglycerides_mgdl']:.0f}", help="Triglycerides")
    with right:
        s = cgm_summary(series)
        st.subheader("CGM summary, 14 days")
        c = st.columns(3)
        c[0].metric("In range", f"{s['tir']:.0%}", help="Time in range 70-180 mg/dL. Consensus target: > 70%")
        c[1].metric("Below 70", f"{s['tbr']:.1%}", help="Time below 70 mg/dL. Consensus target: < 4%")
        c[2].metric("Above 180", f"{s['tar']:.0%}", help="Time above 180 mg/dL. Consensus target: < 25%")
        c = st.columns(3)
        c[0].metric("Mean (mg/dL)", f"{s['mean']:.0f}", help="Mean CGM glucose")
        c[1].metric("GMI", f"{s['gmi']:.1f}%", help="Glucose management indicator: CGM-estimated HbA1c")
        c[2].metric("Variability (CV)", f"{s['cv']:.0%}", help="Stable: <= 36%")
        st.caption(f"CGM data coverage {s['coverage']:.0%}.")
        st.subheader("Alerts over the record")
        c = st.columns(2)
        c[0].metric("Hypoglycaemia alerts", int(tl.hypo_alert.sum()),
                    help="30-min prediction windows at or above the alert threshold")
        c[1].metric("Hyperglycaemia alerts", int(tl.hyper_alert.sum()))
        st.caption("Counts are alerting 30-minute prediction windows, not separate episodes.")


def view_timeline(ctx: TwinContext, twin: PatientTwin, series: pd.DataFrame, tl: pd.DataFrame) -> None:
    hours = st.select_slider("Hours of history shown", options=[6, 12, 24, 48, 72], value=24)
    st.plotly_chart(timeline_figure(series, tl, twin.time, hours, ctx.thresholds), use_container_width=True)
    st.caption("Shaded bands: above 180 mg/dL (high) and below 70 mg/dL (low). Triangles mark alerts (risk at or above "
               "the dashed threshold). The dotted trace after *now* is what actually happened next; the model "
               "had not seen it when it made the prediction.")
    if twin.outcome is not None:
        got = ", ".join(f"{TARGET_NAME[t].lower()}: {'yes' if v else 'no'}" for t, v in twin.outcome.items())
        st.markdown(f"**What actually happened in the next 2 h** (synthetic ground truth): {got}.")


def view_drivers(ctx: TwinContext, twin: PatientTwin) -> None:
    st.markdown("These drivers explain **what the model reacted to** (TreeSHAP), not what physiologically caused "
                "the glucose change. Contributions are in log-odds: right of zero raises the predicted risk.")
    cols = st.columns(2)
    for col, target in zip(cols, ("hypo", "hyper")):
        with col:
            st.markdown(f"#### {TARGET_NAME[target]}: {twin.risk[target]:.0%}"
                        + (" · ALERT" if twin.alert[target] else ""))
            d = explain(ctx, twin, target, top_k=7)
            st.plotly_chart(drivers_figure(d, target), use_container_width=True)
            for line in driver_sentences(d.head(3), target):
                st.markdown(f"- {line}")
            g = group_contributions(ctx, twin, target)
            st.caption("By data stream (log-odds): " + " · ".join(f"{k} {v:+.2f}" for k, v in g.items()))


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
    st.markdown("Change the patient's day around **now** and see two separate answers: what the **risk model** would "
                "predict from the data it would see, and what the **simulated physiology** does over the next 2 h.")
    mode = st.radio("Scenario", ["Late dinner (+2 h)", "Poor sleep night", "Skipped meal", "Extra insulin dose", "Custom"],
                    horizontal=True)
    if mode != "Custom":
        scenarios, note = preset_scenarios(mode, ev, twin.time, bool(twin.profile["on_insulin"]))
        st.caption(note)
    else:
        scenarios = []
        c1, c2, c3 = st.columns(3)
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

    st.markdown("**Scenario:** " + "; ".join(res.descriptions))
    cols = st.columns(2)
    for col, t in zip(cols, ("hypo", "hyper")):
        b, s = res.baseline.risk[t], res.scenario.risk[t]
        flag = " · ALERT" if res.scenario.alert[t] else ""
        pts = int(round(100 * (s - b)))
        if pts:
            col.metric(f"Model: {TARGET_NAME[t].lower()} risk{flag}", f"{s:.0%}", f"{pts:+d} points vs {b:.0%}",
                       delta_color="inverse")
        else:
            col.metric(f"Model: {TARGET_NAME[t].lower()} risk{flag} · no change", f"{s:.0%}")
    if res.unseen_by_model:
        st.warning("The risk model **cannot see** changes after *now* (" + "; ".join(res.unseen_by_model) + "). "
                   "Its risk only reflects past data; the simulator projection below does include them.")
    st.markdown("**Simulated glucose: twin physiology (synthetic simulator)**")
    st.plotly_chart(projection_figure(res, ctx.cfg["thresholds"]), use_container_width=True)
    o = res.simulated_outcome
    st.markdown(
        f"**Simulated next 2 h:** lowest {o['baseline']['true_min']:.0f} → **{o['scenario']['true_min']:.0f}** mg/dL, "
        f"highest {o['baseline']['true_max']:.0f} → **{o['scenario']['true_max']:.0f}** mg/dL. "
        f"CGM low: {'yes' if o['scenario']['hypo'] else 'no'} · CGM high: {'yes' if o['scenario']['hyper'] else 'no'}.")
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
    st.warning(BANNER)
    try:
        ctx = get_context()
    except FileNotFoundError as exc:
        st.error(f"Pipeline outputs not found ({exc.filename}). Run `bash scripts/run_pipeline.sh` first.")
        st.stop()

    st.sidebar.title("Glycaemic digital twin")
    st.sidebar.caption("Type 2 diabetes · 2-hour risk of high and low glucose")
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
    st.markdown(f"## {pid} &nbsp;·&nbsp; {e['age']} y {'M' if e['sex'] == 'M' else 'F'} &nbsp;·&nbsp; {e['regimen']} "
                f"&nbsp;·&nbsp; HbA1c {e['hba1c']}%")
    st.markdown(f"Prediction made at **{fmt_time(twin.time)}**, using the previous 6 hours of data plus the EHR profile.")
    c1, c2 = st.columns(2)
    with c1:
        risk_card(twin, "hypo", ctx.thresholds["hypo"])
    with c2:
        risk_card(twin, "hyper", ctx.thresholds["hyper"])

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
