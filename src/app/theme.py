"""Visual theme for the dashboard: palette, typography, CSS and small HTML components.

Design: calm clinical neutrals, white cards, a deep teal / navy accent. Data colours keep one meaning
everywhere: coral = high glucose (hyperglycaemia), blue = low glucose (hypoglycaemia). Status colours
(red alert, green OK, amber warning) are used only for state, always with an icon and a text label.
"""
from __future__ import annotations

from html import escape

import plotly.graph_objects as go

# --------------------------------------------------------------------------- palette
BG, CARD, LINE, GRID = "#f3f5f8", "#ffffff", "#e2e7ee", "#edf1f6"
INK, INK_2, INK_3 = "#0f1b2a", "#46566a", "#8794a5"
BRAND, BRAND_DEEP, BRAND_SOFT = "#0f766e", "#0b2f3f", "#e6f4f2"
HIGH, HIGH_SOFT = "#eb6834", "#fdeee6"          # hyperglycaemia (validated categorical slot)
LOW, LOW_SOFT = "#2a78d6", "#e7f0fb"            # hypoglycaemia (validated categorical slot)
IN_RANGE = "#3aa17e"
ALERT, ALERT_SOFT, ALERT_LINE = "#b42318", "#fef3f2", "#fecdca"
OK, OK_SOFT, OK_LINE = "#067647", "#ecfdf3", "#abefc6"
WARN, WARN_SOFT, WARN_LINE = "#93370d", "#fffaeb", "#fedf89"
TARGET_COLOR = {"hyper": HIGH, "hypo": LOW}
TARGET_SOFT = {"hyper": HIGH_SOFT, "hypo": LOW_SOFT}

FONT = "Geist, Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
HEAD_FONT = "'Plus Jakarta Sans', Geist, system-ui, sans-serif"


def _one_line(html: str) -> str:
    """Markdown treats indented lines as code blocks, so emit HTML on a single line."""
    # join with a space so wrapped prose keeps its word breaks, then drop the spaces between tags
    return " ".join(p for p in (part.strip() for part in html.splitlines()) if p).replace("> <", "><")


# --------------------------------------------------------------------------- global CSS
# Shadows are tinted with the navy brand hue (not black) and suggest one light source from above.
SHADOW_SM = "0 1px 2px rgba(11,47,63,0.06), 0 2px 8px -4px rgba(11,47,63,0.10)"
SHADOW_MD = "0 1px 2px rgba(11,47,63,0.06), 0 8px 24px -10px rgba(11,47,63,0.18)"
# Fine grain for the hero so the gradient does not read as a flat vector fill.
GRAIN = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/>"
         "<feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.06 0'/></filter>"
         "<rect width='100%' height='100%' filter='url(%23n)'/></svg>\")")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root {{ --bg:{BG}; --card:{CARD}; --line:{LINE}; --ink:{INK}; --ink2:{INK_2}; --ink3:{INK_3}; --brand:{BRAND}; --brand-soft:{BRAND_SOFT};
  color-scheme: light; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg); -webkit-font-smoothing: antialiased; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3.5rem; max-width: 1380px; }}
button, label, [data-baseweb="tab"], [role="radio"] {{ touch-action: manipulation; }}

/* typography (explicit selectors so icon fonts are left alone) */
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, [data-testid="stCaptionContainer"],
label, button, input, textarea, [data-baseweb="select"], [data-baseweb="tab"],
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
[data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] p {{ font-family: {FONT}; }}
.stMarkdown p, .stMarkdown li {{ color: var(--ink); line-height: 1.6; font-size: 0.95rem; max-width: 78ch; text-wrap: pretty; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--ink2) !important; }}
[data-testid="stCaptionContainer"] p, .stCaption p {{ font-size: 0.8rem !important; color: var(--ink2) !important; line-height: 1.5; }}
h1, h2, h3, h4, h5 {{ font-family: {HEAD_FONT} !important; color: var(--ink) !important; letter-spacing: -0.015em; text-wrap: balance; }}
[data-testid="stWidgetLabel"] p {{ font-weight: 600; color: var(--ink2); font-size: 0.85rem; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid var(--line); }}
section[data-testid="stSidebar"] hr {{ border-color: var(--line); }}

/* tabs: hover, focus and active states */
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ height: 46px; padding: 0 18px; border-radius: 10px 10px 0 0; background: transparent;
  transition: background-color .18s ease, color .18s ease; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: 0.95rem; color: var(--ink2); transition: color .18s ease; }}
.stTabs [data-baseweb="tab"]:hover {{ background: rgba(15,118,110,0.06); }}
.stTabs [data-baseweb="tab"]:hover p {{ color: var(--ink); }}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"]:hover {{ background: {CARD}; }}
.stTabs [aria-selected="true"] p {{ color: var(--brand); }}
.stTabs [data-baseweb="tab"]:focus-visible {{ outline: 2px solid var(--brand); outline-offset: -2px; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--brand); height: 3px; border-radius: 3px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* metrics: elevation without an outline; sentence-case labels */
.gt-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 12px; margin: 4px 0 14px; }}
.gt-tile {{ background: {CARD}; border-radius: 14px; padding: 12px 14px 11px; box-shadow: {SHADOW_SM}; min-width: 0; }}
.gt-tile .l {{ font-size: 0.8rem; font-weight: 600; color: var(--ink2); }}
.gt-tile .v {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 1.55rem; color: var(--ink); font-variant-numeric: tabular-nums;
  line-height: 1.2; margin-top: 2px; overflow-wrap: anywhere; }}
.gt-tile .h {{ font-size: 0.72rem; color: var(--ink3); margin-top: 3px; line-height: 1.3; }}
[data-testid="stMetric"] {{ background: {CARD}; border-radius: 14px; padding: 14px 16px 12px; box-shadow: {SHADOW_SM}; }}
[data-testid="stMetricLabel"] p {{ font-size: 0.8rem !important; font-weight: 600; color: var(--ink2) !important; }}
[data-testid="stMetricValue"] {{ font-weight: 700; font-size: 1.65rem; color: var(--ink); font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em; }}

/* charts as cards */
[data-testid="stPlotlyChart"] {{ background: {CARD}; border-radius: 18px; padding: 12px 14px 6px; box-shadow: {SHADOW_MD};
  box-sizing: border-box; overflow: hidden; }}

/* radios as segmented options: hover, pressed, checked, keyboard focus */
div[role="radiogroup"] {{ gap: 6px; }}
div[role="radiogroup"] label[data-baseweb="radio"] {{ background: {CARD}; border: 1px solid var(--line); border-radius: 10px;
  padding: 7px 14px 7px 10px; margin: 0 6px 6px 0; cursor: pointer;
  transition: border-color .18s ease, background-color .18s ease, transform .12s ease; }}
div[role="radiogroup"] label[data-baseweb="radio"]:hover {{ border-color: rgba(15,118,110,0.45); background: #f7fbfa; }}
div[role="radiogroup"] label[data-baseweb="radio"]:active {{ transform: scale(0.98); }}
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{ border-color: var(--brand); background: var(--brand-soft); }}
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:focus-visible) {{ outline: 2px solid var(--brand); outline-offset: 2px; }}
div[role="radiogroup"] label[data-baseweb="radio"] p {{ font-weight: 600; font-size: 0.88rem; color: var(--ink); }}

/* buttons */
.stButton > button {{ border-radius: 10px; font-weight: 600; padding: 0.5rem 1.1rem;
  transition: background-color .18s ease, box-shadow .18s ease, transform .12s ease; }}
.stButton > button[kind="primary"] {{ background: {BRAND}; border-color: {BRAND}; box-shadow: {SHADOW_SM}; }}
.stButton > button[kind="primary"]:hover {{ background: #0c625b; border-color: #0c625b; box-shadow: {SHADOW_MD}; }}
.stButton > button:active {{ transform: translateY(1px); }}
.stButton > button:focus-visible {{ outline: 2px solid {BRAND_DEEP}; outline-offset: 2px; }}

/* native alerts */
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* ---- custom components ---- */
.gt-banner {{ display: flex; gap: 12px; align-items: flex-start; background: {WARN_SOFT}; border: 1px solid {WARN_LINE};
  border-left: 4px solid #f79009; border-radius: 12px; padding: 10px 14px; margin-bottom: 16px; }}
.gt-banner .ico {{ font-size: 1.1rem; line-height: 1.4; }}
.gt-banner .txt {{ color: {WARN}; font-size: 0.86rem; line-height: 1.5; }}
.gt-banner b {{ color: #7a2e0e; }}

.gt-hero {{ position: relative; overflow: hidden; color: #fff; border-radius: 22px; padding: 24px 28px 26px;
  background: radial-gradient(120% 140% at 100% 0%, rgba(45,212,191,0.28) 0%, rgba(45,212,191,0) 55%),
              radial-gradient(90% 120% at 0% 100%, rgba(8,24,36,0.65) 0%, rgba(8,24,36,0) 60%),
              linear-gradient(115deg, {BRAND_DEEP} 0%, #0d4a55 58%, {BRAND} 100%);
  display: flex; justify-content: space-between; gap: 24px; align-items: flex-end; flex-wrap: wrap;
  box-shadow: 0 12px 32px -14px rgba(11,47,63,0.45); margin-bottom: 18px; }}
.gt-hero::after {{ content: ""; position: absolute; inset: 0; background-image: {GRAIN}; pointer-events: none; mix-blend-mode: overlay; }}
.gt-hero > * {{ position: relative; z-index: 1; min-width: 0; }}
.gt-hero .eyebrow {{ font-size: 0.74rem; font-weight: 600; letter-spacing: 0.08em; opacity: 0.78; }}
.gt-hero .name {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: 2.35rem; line-height: 1.05; letter-spacing: -0.025em;
  margin: 6px 0 12px; }}
.gt-hero .chips {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.gt-chip {{ background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.22); border-radius: 8px; padding: 4px 11px;
  font-size: 0.84rem; font-weight: 500; white-space: nowrap; }}
.gt-hero .when {{ text-align: right; }}
.gt-hero .when .label {{ font-size: 0.74rem; font-weight: 600; letter-spacing: 0.08em; opacity: 0.78; }}
.gt-hero .when .value {{ font-family: {HEAD_FONT}; font-size: 1.3rem; font-weight: 700; margin-top: 3px; font-variant-numeric: tabular-nums; }}
.gt-hero .when .sub {{ font-size: 0.8rem; opacity: 0.82; margin-top: 3px; }}

.gt-risk {{ background: {CARD}; border: 1px solid transparent; border-left-width: 5px; border-left-style: solid; border-radius: 18px;
  padding: 16px 20px 15px; box-shadow: {SHADOW_SM}; margin-bottom: 14px; }}
.gt-risk .row {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
.gt-risk .title {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 1.05rem; color: {INK}; display: flex; align-items: center;
  gap: 8px; margin: 0; }}
.gt-risk .dot {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
.gt-risk .rule {{ color: {INK_2}; font-size: 0.82rem; margin-top: 2px; }}
.gt-risk .value {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: 2.9rem; color: {INK}; line-height: 1; letter-spacing: -0.03em;
  margin: 12px 0 12px; font-variant-numeric: tabular-nums; }}
.gt-gauge {{ position: relative; height: 8px; background: {GRID}; border-radius: 99px; }}
.gt-gauge .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; }}
.gt-gauge .thr {{ position: absolute; top: -4px; width: 2px; height: 16px; background: {INK}; border-radius: 2px; }}
.gt-risk .foot {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; color: {INK_2}; font-size: 0.78rem; margin-top: 9px; }}

.gt-pill {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 7px; padding: 3px 10px; font-size: 0.78rem;
  font-weight: 700; letter-spacing: 0.01em; border: 1px solid; white-space: nowrap; }}
.gt-pill.alert {{ background: {ALERT_SOFT}; color: {ALERT}; border-color: {ALERT_LINE}; }}
.gt-pill.ok {{ background: {OK_SOFT}; color: {OK}; border-color: {OK_LINE}; }}
.gt-pill.neutral {{ background: {GRID}; color: {INK_2}; border-color: {LINE}; }}
.gt-pill.warn {{ background: {WARN_SOFT}; color: {WARN}; border-color: {WARN_LINE}; }}
.gt-risk .status {{ color: {INK_2}; font-size: 0.8rem; margin-top: 7px; }}

.gt-section {{ margin: 22px 0 10px; }}
.gt-section .t {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 1.12rem; color: {INK}; letter-spacing: -0.01em;
  text-wrap: balance; margin: 0; }}
.gt-section .s {{ color: {INK_2}; font-size: 0.86rem; margin-top: 3px; max-width: 78ch; text-wrap: pretty; }}

.gt-card {{ background: {CARD}; border-radius: 16px; padding: 6px 18px; box-shadow: {SHADOW_SM}; }}
.gt-table {{ width: 100%; border-collapse: collapse; border: none !important; margin: 0 !important; }}
.gt-table tr {{ background: transparent !important; border: none !important; }}
.gt-table td {{ padding: 10px 4px; border: none !important; border-bottom: 1px solid {GRID} !important; vertical-align: top;
  font-size: 0.9rem; color: {INK}; overflow-wrap: anywhere; }}
.gt-table tr:last-child td {{ border-bottom: none !important; }}
.gt-table td.k {{ color: {INK_2}; font-weight: 600; width: 38%; font-size: 0.84rem; hyphens: auto; word-break: normal; overflow-wrap: normal; }}

.gt-tir {{ display: flex; height: 14px; border-radius: 99px; overflow: hidden; gap: 2px; background: {CARD}; }}
.gt-tir div {{ height: 100%; }}
.gt-tir-legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 9px; font-size: 0.8rem; color: {INK_2}; }}
.gt-tir-legend span i {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
.gt-tir-legend b {{ font-variant-numeric: tabular-nums; }}

.gt-compare {{ background: {CARD}; border-left-width: 5px; border-left-style: solid; border-radius: 18px;
  padding: 14px 18px; box-shadow: {SHADOW_SM}; }}
.gt-compare .t {{ font-weight: 600; color: {INK_2}; font-size: 0.82rem; }}
.gt-compare .v {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: 2.1rem; color: {INK}; margin: 6px 0 4px; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; }}
.gt-compare .v .from {{ color: {INK_3}; font-weight: 700; font-size: 1.3rem; }}
.gt-compare .d {{ font-size: 0.84rem; font-weight: 600; }}

.gt-brand {{ display: flex; gap: 12px; align-items: center; padding: 4px 0 12px; }}
.gt-brand .logo {{ width: 40px; height: 40px; border-radius: 12px; flex: none;
  background: radial-gradient(120% 120% at 100% 0%, rgba(45,212,191,0.45), rgba(45,212,191,0) 60%), linear-gradient(135deg, {BRAND_DEEP}, {BRAND});
  display: flex; align-items: center; justify-content: center; color: #fff; font-family: {HEAD_FONT}; font-weight: 800; font-size: 1rem;
  box-shadow: {SHADOW_SM}; }}
.gt-brand .n {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: 1.05rem; color: {INK}; line-height: 1.2; letter-spacing: -0.01em; }}
.gt-brand .s {{ font-size: 0.78rem; color: {INK_2}; }}

.gt-chiprow {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }}
.gt-chiprow .c {{ background: {CARD}; border: 1px solid var(--line); border-radius: 8px; padding: 3px 10px; font-size: 0.8rem; color: {INK_2}; }}
.gt-chiprow .c b {{ color: {INK}; font-variant-numeric: tabular-nums; }}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }}
}}
</style>
"""


# --------------------------------------------------------------------------- components
def banner_html() -> str:
    return _one_line(f"""
    <div class="gt-banner" role="note" aria-label="Synthetic data notice"><div class="ico" aria-hidden="true">⚠️</div><div class="txt">
    <b>SYNTHETIC DATA. Research prototype, not for clinical use.</b> Every patient, record and sensor trace is simulated,
    and the link between EHR and wearable data is synthetic by design. Risk scores have not been clinically validated.
    </div></div>""")


def brand_html() -> str:
    return _one_line("""
    <div class="gt-brand"><div class="logo" aria-hidden="true">GT</div><div>
    <div class="n">Glycaemic Digital Twin</div><div class="s">Type 2 diabetes · 2-hour risk of high and low glucose</div>
    </div></div>""")


def hero_html(patient_id: str, chips: list[str], when: str, sub: str) -> str:
    chip_html = "".join(f'<span class="gt-chip">{escape(c)}</span>' for c in chips)
    return _one_line(f"""
    <div class="gt-hero"><div>
      <div class="eyebrow">Virtual patient · test set</div>
      <div class="name" role="heading" aria-level="1">{escape(patient_id)}</div>
      <div class="chips">{chip_html}</div>
    </div><div class="when">
      <div class="label">Prediction time</div><div class="value">{escape(when)}</div><div class="sub">{escape(sub)}</div>
    </div></div>""")


def pill_html(text: str, kind: str, icon: str = "") -> str:
    mark = f'<span aria-hidden="true">{icon}</span> ' if icon else ""
    return f'<span class="gt-pill {kind}">{mark}{escape(text)}</span>'


ALERT_STATES = {"alert": ("ALERT RAISED", "alert"), "active": ("Above threshold · alert already raised", "warn"),
                "ok": ("Below threshold", "ok")}


def risk_card_html(target: str, name: str, rule: str, p: float, threshold: float, state: str, status: str,
                   basis: str) -> str:
    """Risk card. `state`: alert (the policy raises an alert now) | active (above threshold, repeat suppressed) | ok."""
    accent, soft = TARGET_COLOR[target], TARGET_SOFT[target]
    label, kind = ALERT_STATES[state]
    icon = ("▲" if target == "hyper" else "▼") if state != "ok" else "✓"
    pill = pill_html(label, kind, icon)
    above = state != "ok"
    card_style = f"border-left-color:{accent};" + (f"border-color:{accent};background:linear-gradient(135deg,{soft} 0%,{CARD} 70%)" if above else "")
    return _one_line(f"""
    <div class="gt-risk" style="{card_style}">
      <div class="row"><div class="title" role="heading" aria-level="2"><span class="dot" aria-hidden="true" style="background:{accent}"></span>{escape(name)} risk</div>{pill}</div>
      <div class="rule">{escape(rule)}</div>
      <div class="value">{p:.0%}</div>
      <div class="gt-gauge" aria-hidden="true"><div class="fill" style="width:{max(p, 0.004) * 100:.1f}%;background:{accent}"></div>
        <div class="thr" style="left:calc({threshold * 100:.1f}% - 1px)"></div></div>
      <div class="foot"><span>Alert threshold <b>{threshold:.0%}</b> (marker)</span>
        <span>{escape(basis)}</span></div>
      <div class="status">{escape(status)}</div>
    </div>""")


def compare_card_html(target: str, name: str, before: float, after: float, alert: bool) -> str:
    pts = int(round(100 * (after - before)))
    if pts > 0:
        delta = f'<span class="d" style="color:{ALERT}">▲ +{pts} points</span>'
    elif pts < 0:
        delta = f'<span class="d" style="color:{OK}">▼ {pts} points</span>'
    else:
        delta = f'<span class="d" style="color:{INK_2}">no change</span>'
    pill = (pill_html("Above alert threshold", "alert", "▲" if target == "hyper" else "▼") if alert
            else pill_html("Below threshold", "ok", "✓"))
    return _one_line(f"""
    <div class="gt-compare" style="border-left-color:{TARGET_COLOR[target]}">
      <div class="row" style="display:flex;justify-content:space-between;align-items:center">
        <div class="t">Model · {escape(name)} risk</div>{pill}</div>
      <div class="v"><span class="from">{before:.0%} →</span> {after:.0%}</div>{delta}
    </div>""")


def section_html(title: str, subtitle: str = "") -> str:
    sub = f'<div class="s">{escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="gt-section"><div class="t" role="heading" aria-level="2">{escape(title)}</div>{sub}</div>'


def table_html(rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<tr><td class="k">{escape(k)}</td><td>{escape(v)}</td></tr>' for k, v in rows)
    return f'<div class="gt-card"><table class="gt-table">{body}</table></div>'


def tir_bar_html(tbr: float, tir: float, tar: float) -> str:
    """Time-in-ranges bar (AGP style): below 70 | 70-180 | above 180."""
    segs = [(tbr, LOW, "Below 70"), (tir, IN_RANGE, "In range 70-180"), (tar, HIGH, "Above 180")]
    bar = "".join(f'<div style="width:{max(v, 0) * 100:.2f}%;background:{c}" title="{n}: {v:.1%}"></div>' for v, c, n in segs if v > 0)
    legend = "".join(f'<span><i style="background:{c}"></i>{n} <b style="color:{INK}">{v:.1%}</b></span>' for v, c, n in segs)
    return f'<div class="gt-card" style="padding:14px 18px"><div class="gt-tir">{bar}</div><div class="gt-tir-legend">{legend}</div></div>'


def tiles_html(items: list[tuple[str, str, str]]) -> str:
    """Number tiles (label, value, hint) in a grid that wraps instead of truncating on narrow screens."""
    return ('<div class="gt-tiles">' + "".join(
        f'<div class="gt-tile" title="{escape(h)}"><div class="l">{escape(k)}</div><div class="v">{escape(v)}</div>'
        + (f'<div class="h">{escape(h)}</div>' if h else "") + "</div>" for k, v, h in items) + "</div>")


def chips_html(items: list[tuple[str, str]]) -> str:
    return '<div class="gt-chiprow">' + "".join(f'<span class="c">{escape(k)} <b>{escape(v)}</b></span>' for k, v in items) + "</div>"


# --------------------------------------------------------------------------- charts
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}


def style_figure(fig: go.Figure, height: int, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
        font=dict(family=FONT, size=12, color=INK_2), margin=dict(l=8, r=12, t=40, b=8),
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font=dict(family=FONT, size=12, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11.5, color=INK_2), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, showline=True, linecolor=LINE, zeroline=False, tickfont=dict(color=INK_2))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, showline=False, zeroline=False, tickfont=dict(color=INK_2),
                     title_font=dict(color=INK_2, size=12))
    return fig
