# app.py
# MomentumAI — Warehouse Pallet & Cart Planning Dashboard
#
# Run:  streamlit run app.py

import math
import joblib
import pandas as pd
import numpy  as np
import streamlit as st
from pathlib import Path

from config     import CONFIG
from preprocess import get_feature_matrix, MODEL_FEATURES

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "MomentumAI",
    page_icon  = "📦",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

  /* ── Reset & base ── */
  html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
  .main { background-color: #070d1a; }
  .block-container { padding: 2rem 2.4rem 3rem; max-width: 1280px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1526;
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  [data-testid="stSidebar"] .block-container { padding: 1.6rem 1.2rem; }
  [data-testid="stSidebarContent"] { padding-top: 0.5rem; }

  /* ── Sidebar logo strip ── */
  .sidebar-brand {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.8rem 0 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.4rem;
  }
  .sidebar-brand-icon {
    font-size: 1.6rem; line-height: 1;
  }
  .sidebar-brand-text {
    font-size: 1.15rem; font-weight: 800;
    background: linear-gradient(90deg,#60a5fa,#34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
  }
  .sidebar-section-label {
    font-size: 0.72rem; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 1.4rem 0 0.6rem;
  }
  .sidebar-divider {
    border: none; border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1rem 0;
  }

  /* ── Page header ── */
  .page-header {
    background: linear-gradient(135deg,#0f1e3d 0%,#0d2651 50%,#0a1f3d 100%);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.6rem;
    position: relative; overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
  }
  .page-header::before {
    content: '';
    position: absolute; top: -40%; right: -10%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
  }
  .header-title {
    font-size: 2.0rem; font-weight: 900; color: #f8fafc;
    letter-spacing: -0.04em; margin: 0 0 0.4rem;
    background: linear-gradient(90deg,#e2e8f0 0%,#93c5fd 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .header-sub {
    font-size: 0.97rem; color: #64748b; line-height: 1.6; max-width: 520px;
  }
  .header-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25);
    padding: 0.25rem 0.75rem; border-radius: 999px;
    font-size: 0.78rem; color: #93c5fd; font-weight: 600;
    margin-top: 0.8rem;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 0;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 10px 10px 0 0;
    color: #475569 !important;
    font-weight: 600; font-size: 0.9rem;
    padding: 0.65rem 1.1rem;
    transition: all 0.18s ease;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #94a3b8 !important; }
  .stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.08) !important;
    color: #60a5fa !important;
    border-bottom: 2px solid #3b82f6 !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.6rem; }

  /* ── Input cards ── */
  .input-card {
    background: #111c32;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
  }
  .input-card-label {
    font-size: 0.75rem; font-weight: 700; color: #3b82f6;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 0.4rem;
  }

  /* ── Results panel ── */
  .results-panel {
    background: #111c32;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
  }

  /* ── Hero result box ── */
  .hero-result {
    background: linear-gradient(135deg,#0c1a38 0%,#112044 100%);
    border: 1px solid rgba(59,130,246,0.22);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
    box-shadow: 0 12px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .hero-result::after {
    content: '';
    position: absolute; bottom: -20px; right: -20px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
  }
  .hero-label {
    font-size: 0.72rem; color: #475569; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem;
  }
  .hero-number {
    font-size: 3rem; font-weight: 900; color: #f1f5f9;
    letter-spacing: -0.05em; line-height: 1;
  }
  .hero-unit {
    font-size: 1.1rem; color: #64748b; font-weight: 500;
    margin-left: 0.3rem;
  }
  .hero-breakdown {
    margin-top: 0.7rem; font-size: 0.84rem; color: #475569;
    display: flex; gap: 0.8rem; flex-wrap: wrap;
  }
  .hero-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px; padding: 0.2rem 0.55rem;
    font-size: 0.78rem; color: #94a3b8;
  }

  /* ── Metric chips ── */
  .metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.75rem; margin-bottom: 1.2rem; }
  .metric-chip {
    background: #0d1829;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    position: relative; overflow: hidden;
  }
  .metric-chip-accent {
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: 12px 12px 0 0;
  }
  .metric-chip-label {
    font-size: 0.72rem; color: #475569; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.35rem;
  }
  .metric-chip-value {
    font-size: 1.55rem; font-weight: 800; color: #e2e8f0;
    letter-spacing: -0.03em; line-height: 1;
  }

  /* ── Risk section ── */
  .risk-section-title {
    font-size: 0.75rem; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 1.2rem 0 0.75rem; display: flex; align-items: center; gap: 0.5rem;
  }
  .risk-cards-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.7rem; }
  .risk-card {
    border-radius: 12px;
    padding: 0.85rem 1rem;
    border-left: 3px solid;
    position: relative;
  }
  .risk-card-green  { background: rgba(16,185,129,0.07); border-color: #10b981; }
  .risk-card-yellow { background: rgba(245,158,11,0.07); border-color: #f59e0b; }
  .risk-card-red    { background: rgba(239,68,68,0.07);  border-color: #ef4444; }
  .risk-card-title  { font-size: 0.8rem; font-weight: 700; margin-bottom: 0.3rem; }
  .risk-title-green  { color: #34d399; }
  .risk-title-yellow { color: #fbbf24; }
  .risk-title-red    { color: #f87171; }
  .risk-card-body   { font-size: 0.78rem; color: #64748b; line-height: 1.45; }

  /* ── Overall risk pill ── */
  .risk-pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.35rem 0.9rem; border-radius: 999px;
    font-size: 0.82rem; font-weight: 700; letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
  }
  .risk-pill-green  { background: rgba(16,185,129,0.12); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
  .risk-pill-yellow { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
  .risk-pill-red    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.30); }

  /* ── Empty state ── */
  .empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 3rem 1.5rem; text-align: center; gap: 0.6rem;
  }
  .empty-icon   { font-size: 2.8rem; opacity: 0.25; }
  .empty-title  { font-size: 1.0rem; color: #334155; font-weight: 600; }
  .empty-body   { font-size: 0.84rem; color: #1e293b; }

  /* ── Run button ── */
  .stButton > button {
    width: 100%; padding: 0.75rem 1.2rem;
    background: linear-gradient(90deg,#2563eb 0%,#0891b2 100%);
    color: white; font-weight: 700; font-size: 0.95rem;
    border: none; border-radius: 12px;
    box-shadow: 0 8px 24px rgba(37,99,235,0.30);
    transition: filter 0.15s, transform 0.12s;
    letter-spacing: 0.01em;
  }
  .stButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); }
  .stButton > button:active { transform: translateY(0); }

  /* ── Streamlit overrides ── */
  div[data-testid="stMetric"] {
    background: #0d1829; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.9rem;
  }
  div[data-testid="stMetricLabel"] > div { color: #475569 !important; font-size: 0.78rem !important; font-weight: 600; }
  div[data-testid="stMetricValue"] > div { color: #e2e8f0 !important; font-weight: 800; }
  .stSlider [data-testid="stSliderThumb"]  { background: #3b82f6; }
  .stSelectbox [data-baseweb="select"]     { background: #0d1829; border-color: rgba(255,255,255,0.1); }
  .stNumberInput input                     { background: #0d1829; border-color: rgba(255,255,255,0.1); color: #e2e8f0; }
  div[data-testid="stCheckbox"]            { gap: 0.4rem; }
  hr { border-color: rgba(255,255,255,0.06); }
  .stDataFrame { border-radius: 12px; overflow: hidden; }
  [data-testid="stDataFrameResizable"] { background: #0d1829; }

  /* ── Alert overrides ── */
  div[data-testid="stAlert"]       { border-radius: 12px; }
  .stSuccess { background: rgba(16,185,129,0.08) !important; border-color: #10b981 !important; }
  .stWarning { background: rgba(245,158,11,0.08) !important; border-color: #f59e0b !important; }
  .stError   { background: rgba(239,68,68,0.08) !important;  border-color: #ef4444 !important; }

  /* ── Info cards (How It Works) ── */
  .info-card {
    background: #111c32; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.4rem 1.6rem;
    box-shadow: 0 8px 28px rgba(0,0,0,0.2);
  }
  .info-card h4 { color: #e2e8f0; font-size: 1.0rem; font-weight: 700; margin-top: 0; }
  .ref-bar {
    background: #111c32; border: 1px solid rgba(59,130,246,0.15);
    border-top: 2px solid #3b82f6;
    border-radius: 12px; padding: 1rem 1.4rem;
    display: flex; gap: 2.5rem; flex-wrap: wrap;
  }
  .ref-item-label { font-size: 0.72rem; color: #475569; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }
  .ref-item-value { font-size: 1.1rem; color: #93c5fd; font-weight: 800; margin-top: 0.1rem; }

  /* ── Weekly day inputs ── */
  .day-input-grid { display: grid; grid-template-columns: repeat(7,1fr); gap: 0.6rem; }
  .day-label { font-size: 0.72rem; color: #3b82f6; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; text-align: center; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
PPP          = CONFIG["packages_per_pallet"]
PPC          = CONFIG["packages_per_cart"]
LEAD_PALLETS = CONFIG["pallet_order_lead_days"]
LEAD_CARTS   = CONFIG["cart_order_lead_days"]
RISK_CFG     = CONFIG["risk"]

SHIFT_TYPE_MAP = {"morning": 0, "afternoon": 1, "night": 2, "peak": 3}
DAYS_OF_WEEK   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

RISK_COLORS = {
    "GREEN":  {"pill": "green",  "icon": "🟢", "label": "LOW RISK"},
    "YELLOW": {"pill": "yellow", "icon": "🟡", "label": "MODERATE RISK"},
    "RED":    {"pill": "red",    "icon": "🔴", "label": "HIGH RISK"},
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = Path("outputs/models/pallet_model.pkl")
    if model_path.exists():
        return joblib.load(str(model_path))
    with st.spinner("⚙️  First run — training MomentumAI model (~20 seconds)…"):
        from train_model import train_pallet_model
        train_pallet_model()
    return joblib.load(str(model_path))

model = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (logic unchanged — only presentation layer changes)
# ══════════════════════════════════════════════════════════════════════════════

def formula_pallets(volume: int) -> int:
    return math.ceil(volume / PPP)

def formula_carts(volume: int) -> int:
    return math.ceil(volume / PPC * 1.08)

def spoilage_buffer(base: int, rate: float, shift_type: str, is_peak: bool) -> int:
    shift_mult = CONFIG["spoilage_by_shift"].get(shift_type, 1.0)
    peak_mult  = CONFIG["peak_season_spoilage_multiplier"] if is_peak else 1.0
    eff_rate   = rate * shift_mult * peak_mult
    return max(CONFIG["min_extra_pallets"], math.ceil(base * eff_rate))

def ai_prediction(volume, shift_type, day_of_week, is_peak,
                  team_size, spoilage_rate, inbound_ratio, has_oversized) -> int:
    row = pd.DataFrame([{
        "target_volume":            volume,
        "shift_type_encoded":       SHIFT_TYPE_MAP[shift_type],
        "day_of_week":              day_of_week,
        "is_weekend":               int(day_of_week >= 5),
        "is_peak_season":           int(is_peak),
        "team_size":                team_size,
        "historical_spoilage_rate": spoilage_rate,
        "inbound_ratio":            inbound_ratio,
        "has_oversized_items":      int(has_oversized),
    }])
    X   = get_feature_matrix(row)
    raw = model.predict(X)[0]
    return max(formula_pallets(volume), math.ceil(raw))

def get_risk(level_key: str, value) -> str:
    if level_key == "stock":
        if value > RISK_CFG["stock_yellow"]:    return "RED"
        if value > RISK_CFG["stock_green"]:     return "YELLOW"
        return "GREEN"
    if level_key == "deadline":
        if value <= RISK_CFG["deadline_red"]:   return "RED"
        if value <= RISK_CFG["deadline_yellow"]: return "YELLOW"
        return "GREEN"
    if level_key == "spoilage":
        if value > RISK_CFG["spoilage_yellow"]: return "RED"
        if value > RISK_CFG["spoilage_green"]:  return "YELLOW"
        return "GREEN"
    return "GREEN"

def overall_risk(levels: list) -> str:
    order = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    return max(levels, key=lambda x: order.get(x, 0))

def metric_chip_html(label, value, accent_color="#3b82f6"):
    return f"""
    <div class="metric-chip">
      <div class="metric-chip-accent" style="background:{accent_color};"></div>
      <div class="metric-chip-label">{label}</div>
      <div class="metric-chip-value">{value}</div>
    </div>"""

def risk_card_html(level, title, body):
    css = f"risk-card risk-card-{level.lower()}"
    title_css = f"risk-title-{level.lower()}"
    return f"""
    <div class="{css}">
      <div class="risk-card-title {title_css}">{title}</div>
      <div class="risk-card-body">{body}</div>
    </div>"""

def risk_pill_html(level):
    rc  = RISK_COLORS[level]
    css = f"risk-pill risk-pill-{rc['pill']}"
    return f'<div class="{css}">{rc["icon"]} &nbsp;{rc["label"]}</div>'


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <span class="sidebar-brand-icon">📦</span>
      <span class="sidebar-brand-text">MomentumAI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">📋 Pallet Status</div>', unsafe_allow_html=True)

    sb_pallet_stock = st.number_input(
        "Current Pallet Stock",
        min_value=0, max_value=5_000, value=250, step=5,
        help="How many pallets you currently have on-hand in your area."
    )
    sb_days_deadline = st.number_input(
        "Days to Order Deadline",
        min_value=0, max_value=14, value=5, step=1,
        help=f"Pallets must be ordered {LEAD_PALLETS} days in advance — this is critical."
    )
    sb_spoilage_pct = st.slider(
        "Area Spoilage Rate",
        min_value=1, max_value=20, value=6, step=1, format="%d%%",
        help="Your area's rolling damaged/unusable pallet percentage."
    )
    sb_spoilage_rate = sb_spoilage_pct / 100.0

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">🛒 Cart Status</div>', unsafe_allow_html=True)

    sb_cart_stock = st.number_input(
        "Current Cart Stock",
        min_value=0, max_value=10_000, value=500, step=10,
        help=f"Carts can be reordered {LEAD_CARTS} day in advance."
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Sidebar live risk summary ──────────────────────────────────────────
    st.markdown('<div class="sidebar-section-label">⚡ Deadline Alert</div>', unsafe_allow_html=True)
    r_dl = get_risk("deadline", sb_days_deadline)
    rc   = RISK_COLORS[r_dl]
    if r_dl == "RED":
        st.error(f"{rc['icon']} **{sb_days_deadline} day(s) left — Order now!**")
    elif r_dl == "YELLOW":
        st.warning(f"{rc['icon']} **{sb_days_deadline} days — Finalise soon**")
    else:
        st.success(f"{rc['icon']} **{sb_days_deadline} days — On track**")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.caption(
        f"📦 {PPP} pkgs/pallet · 🛒 {PPC} pkgs/cart\n\n"
        f"Pallet lead: **{LEAD_PALLETS} days** · Cart lead: **{LEAD_CARTS} day**"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="page-header">
  <div class="header-title">MomentumAI</div>
  <div class="header-sub">
    Intelligent pallet &amp; cart planning for warehouse operations.
    Predict exactly what to order per shift — before the deadline closes.
  </div>
  <div class="header-badge">
    📦 {PPP} packages per pallet &nbsp;·&nbsp;
    🛒 {PPC} packages per cart &nbsp;·&nbsp;
    ⏱ {LEAD_PALLETS}-day pallet lead time
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📦  Shift Planner",
    "📅  Weekly Forecast",
    "ℹ️  How It Works",
])


# ┌─────────────────────────────────────────────────────────────────────────────
# │  TAB 1 — SHIFT PLANNER
# └─────────────────────────────────────────────────────────────────────────────
with tab1:

    input_col, result_col = st.columns([0.95, 1.05], gap="large")

    # ── Left — Shift inputs ──────────────────────────────────────────────────
    with input_col:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-card-label">⚙ Shift Configuration</div>', unsafe_allow_html=True)

        target_volume = st.number_input(
            "📦 Target Volume (packages)",
            min_value=500, max_value=100_000, value=18_000, step=500,
            help="Total packages expected to be processed this shift — the primary input."
        )

        # Live formula preview
        _base_live = formula_pallets(target_volume)
        st.markdown(
            f'<p style="font-size:0.78rem;color:#475569;margin:-0.5rem 0 0.8rem;">'
            f'Formula: ⌈ {target_volume:,} ÷ {PPP} ⌉ = '
            f'<b style="color:#60a5fa">{_base_live} base pallets</b></p>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            shift_type = st.selectbox(
                "Shift Type",
                options=["morning", "afternoon", "night", "peak"],
                format_func=lambda x: {"morning":"🌅 Morning","afternoon":"☀️ Afternoon",
                                        "night":"🌙 Night","peak":"⚡ Peak"}[x],
                help="Night & peak shifts apply higher spoilage multipliers."
            )
        with c2:
            dow_label  = st.selectbox("Day of Week", options=DAYS_OF_WEEK)
            day_of_week = DAYS_OF_WEEK.index(dow_label)

        c3, c4 = st.columns(2)
        with c3:
            team_size = st.number_input("Team Size", min_value=5, max_value=150, value=35, step=1)
        with c4:
            inbound_ratio = st.slider("Inbound %", 0, 100, 55, 5, format="%d%%") / 100.0

        c5, c6 = st.columns(2)
        with c5:
            is_peak_season = st.checkbox(
                "⚡ Peak Season",
                help="Tick for Prime Day, Black Friday, Christmas etc."
            )
        with c6:
            has_oversized = st.checkbox(
                "📏 Oversized Items",
                help="Oversized packages reduce pallet packing efficiency by 8–15%."
            )

        st.markdown('</div>', unsafe_allow_html=True)  # end input-card

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🚀  Run Prediction", key="shift_predict")

    # ── Right — Results ──────────────────────────────────────────────────────
    with result_col:
        st.markdown('<div class="results-panel">', unsafe_allow_html=True)

        if predict_clicked:

            # ── Calculations ──────────────────────────────────────────────
            base   = formula_pallets(target_volume)
            ai_rec = ai_prediction(
                target_volume, shift_type, day_of_week, is_peak_season,
                team_size, sb_spoilage_rate, inbound_ratio, has_oversized
            )
            buf         = spoilage_buffer(base, sb_spoilage_rate, shift_type, is_peak_season)
            total_order = max(ai_rec, base + buf)
            ai_adj      = total_order - base - buf

            carts_needed   = formula_carts(target_volume)
            pallet_deficit = max(0, total_order  - sb_pallet_stock)
            cart_deficit   = max(0, carts_needed - sb_cart_stock)

            pallet_ratio   = total_order / max(sb_pallet_stock, 1)

            r_stock    = get_risk("stock",    pallet_ratio)
            r_deadline = get_risk("deadline", sb_days_deadline)
            r_spoilage = get_risk("spoilage", sb_spoilage_rate)
            r_overall  = overall_risk([r_stock, r_deadline, r_spoilage])
            rc_overall = RISK_COLORS[r_overall]

            # ── Hero result ───────────────────────────────────────────────
            st.markdown(risk_pill_html(r_overall), unsafe_allow_html=True)
            st.markdown(f"""
            <div class="hero-result">
              <div class="hero-label">Total Pallets to Order</div>
              <div>
                <span class="hero-number">{total_order}</span>
                <span class="hero-unit">pallets</span>
              </div>
              <div class="hero-breakdown">
                <span class="hero-chip">📐 Base: {base}</span>
                <span class="hero-chip">🛡 Spoilage: +{buf}</span>
                <span class="hero-chip">🤖 AI adj: {ai_adj:+d}</span>
                <span class="hero-chip">📦 {target_volume:,} pkgs ÷ {PPP}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 4 metric chips ────────────────────────────────────────────
            st.markdown(f"""
            <div class="metric-row">
              {metric_chip_html("Base Pallets",    base,          "#3b82f6")}
              {metric_chip_html("Spoilage Buffer", f"+{buf}",      "#f59e0b")}
              {metric_chip_html("Total to Order",  total_order,   "#10b981")}
              {metric_chip_html("Carts Needed",    carts_needed,  "#6366f1")}
            </div>
            """, unsafe_allow_html=True)

            # ── Shortage banners ──────────────────────────────────────────
            if pallet_deficit > 0:
                st.error(
                    f"⚠️ **Pallet shortage — {pallet_deficit} short.** "
                    f"Stock ({sb_pallet_stock}) can't cover demand ({total_order}). Escalate immediately."
                )
            if cart_deficit > 0:
                st.warning(
                    f"🛒 **Cart shortage — {cart_deficit} short.** "
                    f"Order {cart_deficit} extra carts for tomorrow's shift."
                )

            # ── 3 risk cards ──────────────────────────────────────────────
            st.markdown(
                '<div class="risk-section-title">⚡ Risk Breakdown</div>',
                unsafe_allow_html=True
            )

            stock_msgs = {
                "GREEN":  ("🟢 Stock — OK",
                    f"Stock ({sb_pallet_stock}) covers demand ({total_order}) with headroom."),
                "YELLOW": ("🟡 Stock — Monitor",
                    f"{pallet_ratio*100:.0f}% of stock consumed. Confirm your next order."),
                "RED":    ("🔴 Stock — Critical",
                    f"Demand ({total_order}) nearly depletes stock ({sb_pallet_stock}). "
                    + (f"Short by {pallet_deficit}." if pallet_deficit else "Order today.")),
            }
            deadline_msgs = {
                "GREEN":  ("🟢 Deadline — On Track",
                    f"{sb_days_deadline} days to close. Plenty of time."),
                "YELLOW": ("🟡 Deadline — Plan Now",
                    f"{sb_days_deadline} days left. Finalise your order this week."),
                "RED":    ("🔴 Deadline — Urgent!",
                    f"Only {sb_days_deadline} day(s) left. Place your order TODAY."),
            }
            spoilage_msgs = {
                "GREEN":  ("🟢 Spoilage — Healthy",
                    f"{sb_spoilage_pct}% rate is under control. Buffer of +{buf} is adequate."),
                "YELLOW": ("🟡 Spoilage — Elevated",
                    f"{sb_spoilage_pct}% is above the 7% target. Review handling procedures."),
                "RED":    ("🔴 Spoilage — Critical",
                    f"{sb_spoilage_pct}% is critically high. Immediate review required."),
            }

            st.markdown(f"""
            <div class="risk-cards-grid">
              {risk_card_html(r_stock,    *stock_msgs[r_stock])}
              {risk_card_html(r_deadline, *deadline_msgs[r_deadline])}
              {risk_card_html(r_spoilage, *spoilage_msgs[r_spoilage])}
            </div>
            """, unsafe_allow_html=True)

            # ── Scenario summary (collapsed) ──────────────────────────────
            with st.expander("📋  View scenario summary", expanded=False):
                summary = pd.DataFrame({
                    "Input":  [
                        "Target Volume", "Shift Type",    "Day",
                        "Peak Season",   "Oversized",     "Team Size",
                        "Inbound Ratio", "Spoilage Rate", "Pallet Stock",
                        "Days to Deadline",
                    ],
                    "Value": [
                        f"{target_volume:,} pkgs", shift_type.capitalize(), dow_label,
                        "Yes" if is_peak_season else "No",
                        "Yes" if has_oversized  else "No",
                        team_size,
                        f"{inbound_ratio*100:.0f}%",
                        f"{sb_spoilage_pct}%",
                        sb_pallet_stock,
                        f"{sb_days_deadline} day(s)",
                    ],
                })
                st.dataframe(summary, use_container_width=True, hide_index=True)

        else:
            # ── Empty state ───────────────────────────────────────────────
            _quick_base = formula_pallets(target_volume)
            st.markdown(f"""
            <div class="empty-state">
              <div class="empty-icon">📦</div>
              <div class="empty-title">Ready to predict</div>
              <div class="empty-body">
                Configure your shift on the left and click <strong>Run Prediction</strong>.
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Still show live formula estimate even before running prediction
            _buf_preview  = spoilage_buffer(_quick_base, sb_spoilage_rate, "morning", False)
            _est_total    = _quick_base + _buf_preview
            st.markdown(
                f'<p style="text-align:center;font-size:0.82rem;color:#334155;">'
                f'Live estimate: <b style="color:#60a5fa">{_quick_base}</b> base + '
                f'<b style="color:#f59e0b">{_buf_preview}</b> spoilage buffer = '
                f'<b style="color:#34d399">~{_est_total} pallets</b></p>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)  # end results-panel


# ┌─────────────────────────────────────────────────────────────────────────────
# │  TAB 2 — WEEKLY FORECAST
# └─────────────────────────────────────────────────────────────────────────────
with tab2:

    st.markdown(
        f"Enter expected daily volumes for the week. MomentumAI calculates your "
        f"**total pallet order** (must be placed **{LEAD_PALLETS} days in advance**) "
        f"and daily cart requirements (adjustable **{LEAD_CARTS} day before**).",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Daily volume inputs ──────────────────────────────────────────────────
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-label">📅 Daily Volume Targets (packages)</div>',
                unsafe_allow_html=True)

    day_vols     = {}
    default_vols = [35_000, 38_000, 36_000, 40_000, 42_000, 28_000, 20_000]
    d_cols       = st.columns(7)
    for i, (col, day) in enumerate(zip(d_cols, DAYS_OF_WEEK)):
        with col:
            day_vols[day] = col.number_input(
                day[:3], min_value=0, max_value=150_000,
                value=default_vols[i], step=1_000, key=f"w_{i}"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    wc1, wc2, wc3, wc4 = st.columns(4)
    with wc1:
        weekly_shift = st.selectbox(
            "Predominant Shift",
            ["morning","afternoon","night","peak"],
            format_func=lambda x: x.capitalize(), key="ws"
        )
    with wc2:
        weekly_peak = st.checkbox("Peak Season Week?", key="wp")
    with wc3:
        weekly_team = st.number_input("Avg Team Size", 5, 150, 35, key="wt")
    with wc4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_weekly = st.button("📅  Calculate Weekly Plan", key="w_btn")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if run_weekly:
        rows = []
        total_pallet_order = 0
        for day in DAYS_OF_WEEK:
            vol   = day_vols[day]
            base  = formula_pallets(vol)
            buf   = spoilage_buffer(base, sb_spoilage_rate, weekly_shift, weekly_peak)
            order = base + buf
            carts = formula_carts(vol)
            rows.append({
                "Day": day, "Volume": f"{vol:,}",
                "Base Pallets": base, "Buffer": buf,
                "Total Pallets": order, "Carts": carts,
            })
            total_pallet_order += order

        weekly_df       = pd.DataFrame(rows)
        total_carts_wk  = sum(formula_carts(day_vols[d]) for d in DAYS_OF_WEEK)
        total_vol_wk    = sum(day_vols.values())

        # ── Weekly risk ───────────────────────────────────────────────────
        r_wk_stock    = get_risk("stock",    total_pallet_order / max(sb_pallet_stock, 1))
        r_wk_deadline = get_risk("deadline", sb_days_deadline)
        r_wk_spoilage = get_risk("spoilage", sb_spoilage_rate)
        r_wk_overall  = overall_risk([r_wk_stock, r_wk_deadline, r_wk_spoilage])

        # ── 4 summary metrics ─────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
          {metric_chip_html("Weekly Pallets",  total_pallet_order, "#10b981")}
          {metric_chip_html("Weekly Carts",    total_carts_wk,     "#6366f1")}
          {metric_chip_html("Total Volume",    f"{total_vol_wk:,}", "#3b82f6")}
          {metric_chip_html("Order Deadline",  f"{sb_days_deadline}d", "#f59e0b")}
        </div>
        """, unsafe_allow_html=True)

        # ── Weekly risk pill + action message ─────────────────────────────
        st.markdown(risk_pill_html(r_wk_overall), unsafe_allow_html=True)

        if r_wk_deadline == "RED":
            st.error(
                f"🔴 **Order deadline in {sb_days_deadline} day(s) — Order immediately.** "
                f"Place an order for **{total_pallet_order} pallets** today."
            )
        elif r_wk_deadline == "YELLOW":
            st.warning(
                f"🟡 **{sb_days_deadline} days to deadline.** "
                f"Finalise your order for **{total_pallet_order} pallets** by end of week."
            )
        else:
            st.success(
                f"🟢 On track. Order **{total_pallet_order} pallets** "
                f"before the {LEAD_PALLETS}-day deadline."
            )

        # ── Day-by-day breakdown + chart ──────────────────────────────────
        res_a, res_b = st.columns([1.1, 0.9], gap="large")

        with res_a:
            st.markdown(
                '<div class="input-card-label" style="margin-bottom:0.6rem">📊 Daily Breakdown</div>',
                unsafe_allow_html=True
            )
            st.dataframe(weekly_df, use_container_width=True, hide_index=True)

        with res_b:
            st.markdown(
                '<div class="input-card-label" style="margin-bottom:0.6rem">📈 Volume by Day</div>',
                unsafe_allow_html=True
            )
            chart_df = pd.DataFrame({
                "Day":    [d[:3] for d in DAYS_OF_WEEK],
                "Volume": [day_vols[d] for d in DAYS_OF_WEEK],
            }).set_index("Day")
            st.bar_chart(chart_df, color="#3b82f6", use_container_width=True)

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📅</div>
          <div class="empty-title">Adjust daily volumes above</div>
          <div class="empty-body">Then click <strong>Calculate Weekly Plan</strong> to see your order.</div>
        </div>
        """, unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────────────────────────
# │  TAB 3 — HOW IT WORKS
# └─────────────────────────────────────────────────────────────────────────────
with tab3:

    hw1, hw2 = st.columns(2, gap="large")

    with hw1:
        with st.container():
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 📐 The Core Formula")
            st.markdown(f"""
Every prediction starts with fundamental warehouse math:

> **Base Pallets = ⌈ Target Volume ÷ {PPP} ⌉**

A shift processing **40,000 packages** requires:

> ⌈ 40,000 ÷ {PPP} ⌉ = **{math.ceil(40000/PPP)} pallets**

The AI model refines this using learned patterns across thousands of shifts — shift type, seasonality, team workload, and your area's spoilage history.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 🛡️ Spoilage Buffer System")
            st.markdown(f"""
Damaged/unusable pallets are unavoidable. MomentumAI builds a compound buffer:

| Factor | Multiplier |
|--------|-----------|
| Area spoilage rate | Your input (default {CONFIG['base_spoilage_rate']*100:.0f}%) |
| Night shift | ×{CONFIG['spoilage_by_shift']['night']:.2f} |
| Peak shift | ×{CONFIG['spoilage_by_shift']['peak']:.2f} |
| Peak season | ×{CONFIG['peak_season_spoilage_multiplier']:.2f} extra |
| Safety floor | Always +{CONFIG['min_extra_pallets']} minimum |

> Buffer = max({CONFIG['min_extra_pallets']}, ⌈ Base × Rate × Shift × Season ⌉)
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    with hw2:
        with st.container():
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 🔴🟡🟢 Risk Alert System")
            st.markdown(f"""
Three independent dimensions, evaluated in parallel:

**1. Pallet Stock** — demand vs. current on-hand
- 🟢 Need < {RISK_CFG['stock_green']*100:.0f}% of stock
- 🟡 Need {RISK_CFG['stock_green']*100:.0f}–{RISK_CFG['stock_yellow']*100:.0f}% of stock
- 🔴 Need > {RISK_CFG['stock_yellow']*100:.0f}% (or stock insufficient)

**2. Order Deadline** — days until weekly order closes
- 🟢 > {RISK_CFG['deadline_green']} days
- 🟡 {RISK_CFG['deadline_red']+1}–{RISK_CFG['deadline_green']} days — finalise now
- 🔴 ≤ {RISK_CFG['deadline_red']} days — place order today!

**3. Spoilage Rate** — is your area's damage rate sustainable?
- 🟢 < {RISK_CFG['spoilage_green']*100:.0f}%
- 🟡 {RISK_CFG['spoilage_green']*100:.0f}–{RISK_CFG['spoilage_yellow']*100:.0f}% — review handling
- 🔴 > {RISK_CFG['spoilage_yellow']*100:.0f}% — immediate action

Overall risk = worst of the three.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 🛒 Pallets vs Carts")
            st.markdown(f"""
| | Pallets | Carts |
|-|---------|-------|
| **Capacity** | {PPP} pkgs | {PPC} pkgs |
| **Order lead time** | **{LEAD_PALLETS} days** | **{LEAD_CARTS} day** |
| **If you miss the deadline** | Cannot recover | Usually fixable same day |
| **Primary focus** | ✅ Critical | Secondary |

Pallets are the primary problem because the **{LEAD_PALLETS}-day lead time** leaves no room to recover from a mis-prediction. Carts are provided as a planning aid — they can be corrected almost immediately.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick reference bar ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ref-bar">
      <div>
        <div class="ref-item-label">Pallet Capacity</div>
        <div class="ref-item-value">{PPP} pkgs</div>
      </div>
      <div>
        <div class="ref-item-label">Cart Capacity</div>
        <div class="ref-item-value">{PPC} pkgs</div>
      </div>
      <div>
        <div class="ref-item-label">Pallet Lead Time</div>
        <div class="ref-item-value">{LEAD_PALLETS} days</div>
      </div>
      <div>
        <div class="ref-item-label">Cart Lead Time</div>
        <div class="ref-item-value">{LEAD_CARTS} day</div>
      </div>
      <div>
        <div class="ref-item-label">Default Spoilage</div>
        <div class="ref-item-value">{CONFIG['base_spoilage_rate']*100:.0f}%</div>
      </div>
      <div>
        <div class="ref-item-label">Min Safety Buffer</div>
        <div class="ref-item-value">+{CONFIG['min_extra_pallets']} pallets</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
