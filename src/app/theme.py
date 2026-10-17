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

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
HEAD_FONT = "'Plus Jakarta Sans', Inter, system-ui, sans-serif"


def _one_line(html: str) -> str:
    """Markdown treats indented lines as code blocks, so emit HTML on a single line."""
    return "".join(part.strip() for part in html.splitlines())


# --------------------------------------------------------------------------- global CSS
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root {{ --bg:{BG}; --card:{CARD}; --line:{LINE}; --ink:{INK}; --ink2:{INK_2}; --ink3:{INK_3}; --brand:{BRAND}; --brand-soft:{BRAND_SOFT}; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg); }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1380px; }}

/* typography (explicit selectors so icon fonts are left alone) */
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, [data-testid="stCaptionContainer"],
label, button, input, textarea, [data-baseweb="select"], [data-baseweb="tab"],
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
[data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] p {{ font-family: {FONT}; }}
.stMarkdown p, .stMarkdown li {{ color: var(--ink); line-height: 1.6; font-size: 0.95rem; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--ink2) !important; }}
[data-testid="stCaptionContainer"] p, .stCaption p {{ font-size: 0.8rem !important; color: var(--ink2) !important; line-height: 1.5; }}
h1, h2, h3, h4, h5 {{ font-family: {HEAD_FONT} !important; color: var(--ink) !important; letter-spacing: -0.01em; }}
[data-testid="stWidgetLabel"] p {{ font-weight: 600; color: var(--ink2); font-size: 0.85rem; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid var(--line); }}
section[data-testid="stSidebar"] hr {{ border-color: var(--line); }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ height: 46px; padding: 0 18px; border-radius: 12px 12px 0 0; background: transparent; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: 0.95rem; color: var(--ink2); }}
.stTabs [aria-selected="true"] {{ background: {CARD}; }}
.stTabs [aria-selected="true"] p {{ color: var(--brand); }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--brand); height: 3px; border-radius: 3px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* metrics as cards */
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04); }}
[data-testid="stMetricLabel"] p {{ font-size: 0.74rem !important; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--ink2) !important; }}
[data-testid="stMetricValue"] {{ font-weight: 700; font-size: 1.6rem; color: var(--ink); font-variant-numeric: tabular-nums; }}

/* charts as cards */
[data-testid="stPlotlyChart"] {{ background: {CARD}; border: 1px solid var(--line); border-radius: 16px; padding: 10px 12px 4px;
  box-shadow: 0 1px 3px rgba(16,24,40,0.05); }}

/* radios as pills */
div[role="radiogroup"] {{ gap: 6px; }}
div[role="radiogroup"] label[data-baseweb="radio"] {{ background: {CARD}; border: 1px solid var(--line); border-radius: 999px;
  padding: 6px 14px 6px 10px; margin: 0 6px 6px 0; transition: border-color .15s, background .15s; }}
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {{ border-color: var(--brand); background: var(--brand-soft); }}
div[role="radiogroup"] label[data-baseweb="radio"] p {{ font-weight: 600; font-size: 0.88rem; color: var(--ink); }}

/* native alerts */
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* ---- custom components ---- */
.gt-banner {{ display: flex; gap: 12px; align-items: flex-start; background: {WARN_SOFT}; border: 1px solid {WARN_LINE};
  border-left: 4px solid #f79009; border-radius: 12px; padding: 10px 14px; margin-bottom: 14px; }}
.gt-banner .ico {{ font-size: 1.1rem; line-height: 1.4; }}
.gt-banner .txt {{ color: {WARN}; font-size: 0.86rem; line-height: 1.5; }}
.gt-banner b {{ color: #7a2e0e; }}

.gt-hero {{ background: linear-gradient(120deg, {BRAND_DEEP} 0%, #0d4a55 55%, {BRAND} 100%); color: #fff; border-radius: 20px;
  padding: 22px 26px; display: flex; justify-content: space-between; gap: 24px; align-items: flex-end; flex-wrap: wrap;
  box-shadow: 0 8px 24px rgba(11,47,63,0.18); margin-bottom: 16px; }}
.gt-hero .eyebrow {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.75; }}
.gt-hero .name {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: 2.1rem; line-height: 1.15; margin: 4px 0 10px; }}
.gt-hero .chips {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.gt-chip {{ background: rgba(255,255,255,0.13); border: 1px solid rgba(255,255,255,0.24); border-radius: 999px; padding: 4px 12px;
  font-size: 0.84rem; font-weight: 500; white-space: nowrap; }}
.gt-hero .when {{ text-align: right; }}
.gt-hero .when .label {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.75; }}
.gt-hero .when .value {{ font-family: {HEAD_FONT}; font-size: 1.25rem; font-weight: 700; margin-top: 2px; }}
.gt-hero .when .sub {{ font-size: 0.8rem; opacity: 0.8; margin-top: 2px; }}

.gt-risk {{ background: {CARD}; border: 1px solid var(--line); border-left-width: 5px; border-left-style: solid; border-radius: 16px;
  padding: 16px 20px 14px; box-shadow: 0 1px 3px rgba(16,24,40,0.05); margin-bottom: 14px; }}
.gt-risk .row {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
.gt-risk .title {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 1.02rem; color: {INK}; display: flex; align-items: center; gap: 8px; }}
.gt-risk .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.gt-risk .rule {{ color: {INK_2}; font-size: 0.82rem; margin-top: 2px; }}
.gt-risk .value {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: 2.7rem; color: {INK}; line-height: 1.05; margin: 10px 0 10px;
  font-variant-numeric: tabular-nums; }}
.gt-gauge {{ position: relative; height: 8px; background: {GRID}; border-radius: 99px; }}
.gt-gauge .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; }}
.gt-gauge .thr {{ position: absolute; top: -4px; width: 2px; height: 16px; background: {INK}; border-radius: 2px; }}
.gt-risk .foot {{ display: flex; justify-content: space-between; color: {INK_2}; font-size: 0.78rem; margin-top: 8px; }}

.gt-pill {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 3px 11px; font-size: 0.78rem;
  font-weight: 700; letter-spacing: 0.02em; border: 1px solid; white-space: nowrap; }}
.gt-pill.alert {{ background: {ALERT_SOFT}; color: {ALERT}; border-color: {ALERT_LINE}; }}
.gt-pill.ok {{ background: {OK_SOFT}; color: {OK}; border-color: {OK_LINE}; }}
.gt-pill.neutral {{ background: {GRID}; color: {INK_2}; border-color: {LINE}; }}

.gt-section {{ margin: 18px 0 8px; }}
.gt-section .t {{ font-family: {HEAD_FONT}; font-weight: 700; font-size: 1.08rem; color: {INK}; }}
.gt-section .s {{ color: {INK_2}; font-size: 0.86rem; margin-top: 2px; }}

.gt-card {{ background: {CARD}; border: 1px solid var(--line); border-radius: 16px; padding: 6px 18px;
  box-shadow: 0 1px 3px rgba(16,24,40,0.05); }}
.gt-table {{ width: 100%; border-collapse: collapse; border: none !important; margin: 0 !important; }}
.gt-table tr {{ background: transparent !important; border: none !important; }}
.gt-table td {{ padding: 10px 4px; border: none !important; border-bottom: 1px solid {GRID} !important; vertical-align: top;
  font-size: 0.9rem; color: {INK}; }}
.gt-table tr:last-child td {{ border-bottom: none !important; }}
.gt-table td.k {{ color: {INK_2}; font-weight: 600; width: 38%; font-size: 0.84rem; }}

.gt-tir {{ display: flex; height: 14px; border-radius: 99px; overflow: hidden; gap: 2px; background: {CARD}; }}
.gt-tir div {{ height: 100%; }}
.gt-tir-legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 0.8rem; color: {INK_2}; }}
.gt-tir-legend span i {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}

.gt-compare {{ background: {CARD}; border: 1px solid var(--line); border-left-width: 5px; border-left-style: solid; border-radius: 16px;
  padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,24,40,0.05); }}
.gt-compare .t {{ font-weight: 700; color: {INK_2}; font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; }}
.gt-compare .v {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: 2rem; color: {INK}; margin: 6px 0 4px;
  font-variant-numeric: tabular-nums; }}
.gt-compare .v .from {{ color: {INK_3}; font-weight: 700; font-size: 1.3rem; }}
.gt-compare .d {{ font-size: 0.84rem; font-weight: 600; }}

.gt-brand {{ display: flex; gap: 12px; align-items: center; padding: 4px 0 10px; }}
.gt-brand .logo {{ width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, {BRAND_DEEP}, {BRAND});
  display: flex; align-items: center; justify-content: center; color: #fff; font-family: {HEAD_FONT}; font-weight: 800; font-size: 1rem; }}
.gt-brand .n {{ font-family: {HEAD_FONT}; font-weight: 800; font-size: 1.05rem; color: {INK}; line-height: 1.2; }}
.gt-brand .s {{ font-size: 0.78rem; color: {INK_2}; }}

.gt-chiprow {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }}
.gt-chiprow .c {{ background: {CARD}; border: 1px solid var(--line); border-radius: 999px; padding: 3px 10px; font-size: 0.8rem; color: {INK_2}; }}
.gt-chiprow .c b {{ color: {INK}; font-variant-numeric: tabular-nums; }}
</style>
"""


# --------------------------------------------------------------------------- components
def banner_html() -> str:
    return _one_line(f"""
    <div class="gt-banner"><div class="ico">⚠️</div><div class="txt">
    <b>SYNTHETIC DATA. Research prototype, not for clinical use.</b> Every patient, record and sensor trace is simulated,
    and the link between EHR and wearable data is synthetic by design. Risk scores have not been clinically validated.
    </div></div>""")


def brand_html() -> str:
    return _one_line("""
    <div class="gt-brand"><div class="logo">GT</div><div>
    <div class="n">Glycaemic Digital Twin</div><div class="s">Type 2 diabetes · 2-hour risk of high and low glucose</div>
    </div></div>""")


def hero_html(patient_id: str, chips: list[str], when: str, sub: str) -> str:
    chip_html = "".join(f'<span class="gt-chip">{escape(c)}</span>' for c in chips)
    return _one_line(f"""
    <div class="gt-hero"><div>
      <div class="eyebrow">Virtual patient · test set</div>
      <div class="name">{escape(patient_id)}</div>
      <div class="chips">{chip_html}</div>
    </div><div class="when">
      <div class="label">Prediction time</div><div class="value">{escape(when)}</div><div class="sub">{escape(sub)}</div>
    </div></div>""")


def pill_html(text: str, kind: str, icon: str = "") -> str:
    return f'<span class="gt-pill {kind}">{icon} {escape(text)}</span>'


def risk_card_html(target: str, name: str, rule: str, p: float, threshold: float, alert: bool) -> str:
    accent, soft = TARGET_COLOR[target], TARGET_SOFT[target]
    pill = pill_html("ALERT", "alert", "▲" if target == "hyper" else "▼") if alert else pill_html("Below threshold", "ok", "✓")
    card_style = f"border-left-color:{accent};" + (f"border-color:{accent};background:linear-gradient(135deg,{soft} 0%,{CARD} 70%)" if alert else "")
    return _one_line(f"""
    <div class="gt-risk" style="{card_style}">
      <div class="row"><div class="title"><span class="dot" style="background:{accent}"></span>{escape(name)} risk</div>{pill}</div>
      <div class="rule">{escape(rule)}</div>
      <div class="value">{p:.0%}</div>
      <div class="gt-gauge"><div class="fill" style="width:{max(p, 0.004) * 100:.1f}%;background:{accent}"></div>
        <div class="thr" style="left:calc({threshold * 100:.1f}% - 1px)"></div></div>
      <div class="foot"><span>Alert threshold <b>{threshold:.0%}</b> (marker)</span>
        <span>5% false-alert rate on validation patients</span></div>
    </div>""")


def compare_card_html(target: str, name: str, before: float, after: float, alert: bool) -> str:
    pts = int(round(100 * (after - before)))
    if pts > 0:
        delta = f'<span class="d" style="color:{ALERT}">▲ +{pts} points</span>'
    elif pts < 0:
        delta = f'<span class="d" style="color:{OK}">▼ {pts} points</span>'
    else:
        delta = f'<span class="d" style="color:{INK_2}">no change</span>'
    pill = pill_html("ALERT", "alert", "▲" if target == "hyper" else "▼") if alert else pill_html("Below threshold", "ok", "✓")
    return _one_line(f"""
    <div class="gt-compare" style="border-left-color:{TARGET_COLOR[target]}">
      <div class="row" style="display:flex;justify-content:space-between;align-items:center">
        <div class="t">Model · {escape(name)} risk</div>{pill}</div>
      <div class="v"><span class="from">{before:.0%} →</span> {after:.0%}</div>{delta}
    </div>""")


def section_html(title: str, subtitle: str = "") -> str:
    sub = f'<div class="s">{escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="gt-section"><div class="t">{escape(title)}</div>{sub}</div>'


def table_html(rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<tr><td class="k">{escape(k)}</td><td>{escape(v)}</td></tr>' for k, v in rows)
    return f'<div class="gt-card"><table class="gt-table">{body}</table></div>'


def tir_bar_html(tbr: float, tir: float, tar: float) -> str:
    """Time-in-ranges bar (AGP style): below 70 | 70-180 | above 180."""
    segs = [(tbr, LOW, "Below 70"), (tir, IN_RANGE, "In range 70-180"), (tar, HIGH, "Above 180")]
    bar = "".join(f'<div style="width:{max(v, 0) * 100:.2f}%;background:{c}" title="{n}: {v:.1%}"></div>' for v, c, n in segs if v > 0)
    legend = "".join(f'<span><i style="background:{c}"></i>{n} <b style="color:{INK}">{v:.1%}</b></span>' for v, c, n in segs)
    return f'<div class="gt-card" style="padding:14px 18px"><div class="gt-tir">{bar}</div><div class="gt-tir-legend">{legend}</div></div>'


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
