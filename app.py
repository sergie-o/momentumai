# app.py
# MomentumAI — Warehouse Pallet & Cart Planning Dashboard
#
# Run:  streamlit run app.py
#
# To use your own logo: save it as  momentum_logo.png  in this folder.
# The app will automatically use it in the header and sidebar.

import math
import joblib
import pandas as pd
import numpy  as np
import streamlit as st
from pathlib import Path
from datetime import date

from config     import CONFIG
from preprocess import get_feature_matrix, MODEL_FEATURES
from db         import (init_db, log_shift, get_history,
                        get_real_spoilage_rate, get_summary_stats, delete_shift)

# Initialise database on every startup (safe — creates tables only if missing)
init_db()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title            = "MomentumAI",
    page_icon             = "📦",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# LOGO
# Priority 1: momentum_logo.png in this folder  (your actual PNG)
# Priority 2: built-in SVG fallback encoded as base64 (always works)
#
# To add your logo: copy momentum_logo.png into the same folder as app.py
# ══════════════════════════════════════════════════════════════════════════════
import base64

_SVG_RAW = """<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <!-- 3D Pallet top face -->
  <polygon points="10,72 70,72 80,62 20,62" fill="#d97706"/>
  <!-- 3D Pallet front face -->
  <rect x="10" y="72" width="60" height="10" rx="1" fill="#b45309"/>
  <!-- Pallet legs -->
  <rect x="13" y="82" width="10" height="8" rx="1" fill="#92400e"/>
  <rect x="34" y="82" width="10" height="8" rx="1" fill="#92400e"/>
  <rect x="55" y="82" width="10" height="8" rx="1" fill="#92400e"/>
  <!-- Pallet bottom board -->
  <rect x="10" y="88" width="60" height="5" rx="1" fill="#b45309"/>
  <!-- Deck slat lines -->
  <line x1="28" y1="62" x2="28" y2="72" stroke="#92400e" stroke-width="1.5"/>
  <line x1="46" y1="62" x2="46" y2="72" stroke="#92400e" stroke-width="1.5"/>
  <line x1="64" y1="62" x2="64" y2="72" stroke="#92400e" stroke-width="1.5"/>
  <!-- Upward arrow body (orange base) -->
  <polygon points="44,18 56,18 56,58 44,58" fill="#f97316"/>
  <!-- Arrow gradient overlay (yellow in middle) -->
  <polygon points="44,30 56,30 56,45 44,45" fill="#eab308"/>
  <!-- Arrow head -->
  <polygon points="35,22 50,4 65,22" fill="#65a30d"/>
  <!-- Brain outline (right side) -->
  <path d="M72,30 Q68,24 70,18 Q74,10 82,12 Q88,8 94,14 Q102,12 106,20 Q112,22 110,32 Q114,40 108,46 Q106,54 98,52 Q94,58 86,56 Q80,60 76,54 Q68,52 68,44 Q64,38 72,30 Z"
        fill="#1d4ed8" opacity="0.15"/>
  <path d="M72,30 Q68,24 70,18 Q74,10 82,12 Q88,8 94,14 Q102,12 106,20 Q112,22 110,32 Q114,40 108,46 Q106,54 98,52 Q94,58 86,56 Q80,60 76,54 Q68,52 68,44 Q64,38 72,30 Z"
        fill="none" stroke="#2563eb" stroke-width="2.5"/>
  <!-- Circuit lines -->
  <line x1="80" y1="22" x2="96" y2="22" stroke="#2563eb" stroke-width="2" stroke-linecap="round"/>
  <line x1="76" y1="32" x2="100" y2="32" stroke="#2563eb" stroke-width="2" stroke-linecap="round"/>
  <line x1="78" y1="42" x2="98" y2="42" stroke="#2563eb" stroke-width="2" stroke-linecap="round"/>
  <!-- Circuit dots -->
  <circle cx="78"  cy="22" r="3.5" fill="#2563eb"/>
  <circle cx="100" cy="32" r="3.5" fill="#2563eb"/>
  <circle cx="78"  cy="42" r="3.5" fill="#2563eb"/>
  <circle cx="100" cy="22" r="2.5" fill="#3b82f6"/>
  <circle cx="76"  cy="32" r="2.5" fill="#3b82f6"/>
  <circle cx="100" cy="42" r="2.5" fill="#3b82f6"/>
</svg>"""

# Encode SVG as base64 — works reliably in all browsers via <img src="data:...">
_SVG_B64    = base64.b64encode(_SVG_RAW.encode()).decode()
LOGO_B64_URI = f"data:image/svg+xml;base64,{_SVG_B64}"
LOGO_PNG     = Path("momentum_logo.png")

def get_logo_src(png_width=None):
    """Return src string for the logo — PNG if available, SVG base64 otherwise."""
    if LOGO_PNG.exists():
        with open(LOGO_PNG, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    return LOGO_B64_URI

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  (light theme built on top of config.toml baseline)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
  .main                      { background-color: #eef2ff; }
  .block-container           { padding: 1.8rem 2.2rem 3rem; max-width: 1300px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"]  { background: #0f1e3d !important; }
  [data-testid="stSidebar"] .block-container { padding: 1.4rem 1.1rem; }
  /* Force all sidebar text white */
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] div,
  [data-testid="stSidebar"] span { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; }
  /* Sidebar inputs */
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] [data-baseweb="input"] {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #f1f5f9 !important;
  }
  [data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #f1f5f9 !important;
  }

  /* ── Sidebar brand strip ── */
  .sb-brand {
    display: flex; align-items: center; gap: 0.65rem;
    padding: 0.6rem 0 1.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.3rem;
  }
  .sb-brand-name {
    font-size: 1.15rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(90deg,#93c5fd,#6ee7b7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .sb-section {
    font-size: 0.7rem; font-weight: 700; color: #475569 !important;
    text-transform: uppercase; letter-spacing: 0.1em; margin: 1.3rem 0 0.5rem;
  }
  .sb-divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1rem 0; }

  /* ── Page header banner (stays dark for brand contrast) ── */
  .page-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 60%, #1d4ed8 100%);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.6rem;
    display: flex; align-items: center; gap: 1.5rem;
    box-shadow: 0 12px 40px rgba(30,58,138,0.30), 0 2px 8px rgba(0,0,0,0.12);
    position: relative; overflow: hidden;
  }
  .page-header::after {
    content:''; position:absolute; top:-40%; right:-5%;
    width:300px; height:300px;
    background: radial-gradient(circle,rgba(255,255,255,0.06) 0%,transparent 70%);
    pointer-events:none;
  }
  .header-logo { flex-shrink: 0; }
  .header-logo img, .header-logo svg { width: 60px; height: 60px; }
  .header-text { flex: 1; }
  .header-title {
    font-size: 1.9rem; font-weight: 900; color: #ffffff;
    letter-spacing: -0.04em; margin: 0 0 0.3rem;
  }
  .header-sub  { font-size: 0.95rem; color: rgba(255,255,255,0.65); line-height: 1.5; }
  .header-badge {
    display: inline-flex; align-items: center; gap: 0.4rem; margin-top: 0.7rem;
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.18);
    padding: 0.25rem 0.8rem; border-radius: 999px;
    font-size: 0.76rem; color: rgba(255,255,255,0.80); font-weight: 600;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid #e2e8f0;
    gap: 0.2rem;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important; border-radius: 8px 8px 0 0;
    color: #94a3b8 !important;
    font-weight: 600; font-size: 0.88rem; padding: 0.6rem 1.1rem;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #475569 !important; background: #f1f5f9 !important; }
  .stTabs [aria-selected="true"] {
    color: #2563eb !important;
    background: #eff6ff !important;
    border-bottom: 2px solid #2563eb !important;
    margin-bottom: -2px;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

  /* ── Cards ── */
  .card {
    background: #ffffff;
    border: 1px solid rgba(99,102,241,0.10);
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 2px 12px rgba(99,102,241,0.06), 0 1px 4px rgba(0,0,0,0.04);
  }
  .card-label {
    font-size: 0.72rem; font-weight: 700; color: #2563eb;
    text-transform: uppercase; letter-spacing: 0.09em;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 0.35rem;
  }

  /* ── Live formula preview ── */
  .formula-preview {
    font-size: 0.78rem; color: #94a3b8; margin: -0.5rem 0 0.9rem;
    display: flex; align-items: center; gap: 0.3rem; flex-wrap: wrap;
  }
  .formula-result { color: #2563eb; font-weight: 700; }

  /* ── Hero result ── */
  .hero {
    background: linear-gradient(135deg,#eff6ff 0%,#e0e7ff 100%);
    border: 1.5px solid rgba(37,99,235,0.18);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
    position: relative; overflow: hidden;
  }
  .hero::after {
    content:''; position:absolute; bottom:-16px; right:-16px;
    width:100px; height:100px;
    background: radial-gradient(circle,rgba(99,102,241,0.08) 0%,transparent 70%);
  }
  .hero-label    { font-size:0.7rem; color:#6366f1; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem; }
  .hero-num      { font-size:3rem; font-weight:900; color:#0f172a; letter-spacing:-0.05em; line-height:1.05; }
  .hero-unit     { font-size:1.1rem; color:#64748b; font-weight:500; margin-left:0.25rem; }
  .hero-chips    { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.65rem; }
  .hero-chip     {
    background: rgba(37,99,235,0.07); border:1px solid rgba(37,99,235,0.14);
    border-radius:6px; padding:0.18rem 0.55rem;
    font-size:0.75rem; color:#1e40af; font-weight:500;
  }

  /* ── Metric chips ── */
  .metric-row  { display:grid; grid-template-columns:repeat(4,1fr); gap:0.7rem; margin-bottom:1.1rem; }
  .metric-chip {
    background: #f8faff;
    border: 1px solid rgba(99,102,241,0.12);
    border-top: 3px solid;
    border-radius: 12px; padding: 0.85rem 1rem;
  }
  .mc-label  { font-size:0.68rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.3rem; }
  .mc-value  { font-size:1.5rem; font-weight:800; color:#0f172a; letter-spacing:-0.03em; line-height:1; }

  /* ── Risk pill ── */
  .risk-pill {
    display:inline-flex; align-items:center; gap:0.4rem;
    padding:0.3rem 0.85rem; border-radius:999px;
    font-size:0.8rem; font-weight:700; letter-spacing:0.02em; margin-bottom:0.5rem;
  }
  .rp-green  { background:#f0fdf4; color:#15803d; border:1px solid rgba(22,163,74,0.25); }
  .rp-yellow { background:#fefce8; color:#92400e; border:1px solid rgba(202,138,4,0.25);  }
  .rp-red    { background:#fff1f2; color:#9f1239; border:1px solid rgba(225,29,72,0.25);  }

  /* ── Risk cards ── */
  .risk-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:0.65rem; }
  .risk-card {
    border-radius:12px; padding:0.85rem 1rem;
    border-left:3px solid; border-top:none; border-right:none; border-bottom:none;
    border-style: solid;
  }
  .rc-green  { background:#f0fdf4; border-color:#16a34a; }
  .rc-yellow { background:#fefce8; border-color:#d97706; }
  .rc-red    { background:#fff1f2; border-color:#dc2626; }
  .rc-title  { font-size:0.8rem; font-weight:700; margin-bottom:0.3rem; }
  .rct-green  { color:#15803d; }
  .rct-yellow { color:#92400e; }
  .rct-red    { color:#9f1239; }
  .rc-body   { font-size:0.76rem; color:#64748b; line-height:1.45; }

  /* ── Section divider label ── */
  .section-label {
    font-size:0.72rem; font-weight:700; color:#94a3b8;
    text-transform:uppercase; letter-spacing:0.1em;
    display:flex; align-items:center; gap:0.4rem;
    margin:1.1rem 0 0.65rem;
  }

  /* ── Empty state ── */
  .empty {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; padding:2.5rem 1rem; text-align:center; gap:0.5rem;
  }
  .empty-icon  { font-size:2.6rem; opacity:0.2; }
  .empty-title { font-size:1.0rem; color:#94a3b8; font-weight:600; }
  .empty-body  { font-size:0.82rem; color:#cbd5e1; }

  /* ── Run button ── */
  .stButton > button {
    width:100%; padding:0.75rem 1.2rem;
    background: linear-gradient(90deg,#2563eb 0%,#0891b2 100%);
    color:white !important; font-weight:700; font-size:0.94rem;
    border:none; border-radius:12px;
    box-shadow:0 6px 20px rgba(37,99,235,0.25);
    transition:filter 0.15s, transform 0.12s;
  }
  .stButton > button:hover  { filter:brightness(1.07); transform:translateY(-1px); }
  .stButton > button:active { transform:translateY(0); }

  /* ── Input overrides (light theme) ── */
  input[type="number"], input[type="text"] {
    background: #f8fafc !important; border-color: #e2e8f0 !important;
    color: #0f172a !important; border-radius: 8px !important;
  }
  [data-baseweb="select"] > div:first-child {
    background: #f8fafc !important; border-color: #e2e8f0 !important;
  }
  .stCheckbox label span { color: #334155 !important; }

  /* ── Streamlit metric override ── */
  div[data-testid="stMetric"] {
    background: #f8faff; border: 1px solid rgba(99,102,241,0.12);
    border-radius: 12px; padding: 0.9rem;
  }
  div[data-testid="stMetricLabel"] > div { color: #64748b !important; font-size:0.76rem !important; font-weight:600; }
  div[data-testid="stMetricValue"] > div { color: #0f172a !important; font-weight:800; }

  /* ── Alerts ── */
  [data-testid="stAlert"]  { border-radius:12px; }
  div[data-baseweb="notification"][kind="positive"] { background:#f0fdf4; }
  div[data-baseweb="notification"][kind="warning"]  { background:#fefce8; }
  div[data-baseweb="notification"][kind="negative"] { background:#fff1f2; }

  /* ── Dataframe ── */
  .stDataFrame { border-radius:12px; overflow:hidden; }

  /* ── Info cards (How It Works) ── */
  .info-card { background:#ffffff; border:1px solid rgba(99,102,241,0.10); border-radius:16px; padding:1.4rem 1.6rem; box-shadow:0 2px 12px rgba(99,102,241,0.06); }
  .info-card h4 { color:#0f172a; font-size:1.0rem; font-weight:700; margin-top:0; }

  /* ── Reference bar ── */
  .ref-bar {
    background:#ffffff; border:1px solid rgba(99,102,241,0.12);
    border-top:3px solid #2563eb;
    border-radius:14px; padding:1.1rem 1.5rem;
    display:flex; gap:2.5rem; flex-wrap:wrap;
    box-shadow:0 2px 12px rgba(99,102,241,0.06);
  }
  .ref-label { font-size:0.7rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; }
  .ref-value { font-size:1.05rem; color:#2563eb; font-weight:800; margin-top:0.1rem; }

  hr { border-color: #e2e8f0; }
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


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    mp = Path("outputs/models/pallet_model.pkl")
    if mp.exists():
        return joblib.load(str(mp))
    with st.spinner("⚙️  First run — training MomentumAI (~20 s)…"):
        from train_model import train_pallet_model
        train_pallet_model()
    return joblib.load(str(mp))

model = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def formula_pallets(volume: int) -> int:
    return math.ceil(volume / PPP)

def formula_carts(volume: int) -> int:
    return math.ceil(volume / PPC * 1.08)

def spoilage_buffer(base: int, rate: float, shift_type: str, is_peak: bool) -> int:
    sm  = CONFIG["spoilage_by_shift"].get(shift_type, 1.0)
    pm  = CONFIG["peak_season_spoilage_multiplier"] if is_peak else 1.0
    return max(CONFIG["min_extra_pallets"], math.ceil(base * rate * sm * pm))

def ai_prediction(volume, shift_type, dow, is_peak,
                  team_size, spoilage_rate, inbound_ratio, has_oversized) -> int:
    row = pd.DataFrame([{
        "target_volume":            volume,
        "shift_type_encoded":       SHIFT_TYPE_MAP[shift_type],
        "day_of_week":              dow,
        "is_weekend":               int(dow >= 5),
        "is_peak_season":           int(is_peak),
        "team_size":                team_size,
        "historical_spoilage_rate": spoilage_rate,
        "inbound_ratio":            inbound_ratio,
        "has_oversized_items":      int(has_oversized),
    }])
    raw = model.predict(get_feature_matrix(row))[0]
    return max(formula_pallets(volume), math.ceil(raw))

def get_risk(key: str, value) -> str:
    if key == "stock":
        if value > RISK_CFG["stock_yellow"]:     return "RED"
        if value > RISK_CFG["stock_green"]:      return "YELLOW"
        return "GREEN"
    if key == "deadline":
        if value <= RISK_CFG["deadline_red"]:    return "RED"
        if value <= RISK_CFG["deadline_yellow"]: return "YELLOW"
        return "GREEN"
    if key == "spoilage":
        if value > RISK_CFG["spoilage_yellow"]:  return "RED"
        if value > RISK_CFG["spoilage_green"]:   return "YELLOW"
        return "GREEN"
    return "GREEN"

def worst_risk(levels: list) -> str:
    order = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    return max(levels, key=lambda x: order.get(x, 0))

# ── HTML builders ──────────────────────────────────────────────────────────────
def metric_chip(label, value, border_color):
    return f"""
    <div class="metric-chip" style="border-top-color:{border_color}">
      <div class="mc-label">{label}</div>
      <div class="mc-value">{value}</div>
    </div>"""

def risk_pill(level):
    icons = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    labels = {"GREEN": "LOW RISK", "YELLOW": "MODERATE RISK", "RED": "HIGH RISK"}
    css = {"GREEN": "rp-green", "YELLOW": "rp-yellow", "RED": "rp-red"}
    return f'<div class="risk-pill {css[level]}">{icons[level]}&nbsp;{labels[level]}</div>'

def risk_card(level, title, body):
    return f"""
    <div class="risk-card rc-{level.lower()}">
      <div class="rc-title rct-{level.lower()}">{title}</div>
      <div class="rc-body">{body}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand strip with logo (base64 encoded — works in all browsers)
    st.markdown(f"""
    <div class="sb-brand">
      <img src="{get_logo_src()}" style="width:42px;height:42px;object-fit:contain;flex-shrink:0;">
      <span class="sb-brand-name">MomentumAI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">📦 Pallet Status</div>', unsafe_allow_html=True)

    sb_pallet_stock = st.number_input(
        "Current Pallet Stock", min_value=0, max_value=5_000, value=250, step=5,
        help="How many pallets you currently have on-hand."
    )
    sb_days_deadline = st.number_input(
        "Days to Order Deadline", min_value=0, max_value=14, value=5, step=1,
        help=f"Pallets must be ordered {LEAD_PALLETS} days in advance."
    )
    # Use real spoilage rate from logged shifts if enough data exists
    _real_rate, _n_shifts = get_real_spoilage_rate(min_shifts=3)
    if _real_rate is not None:
        _default_pct = max(1, min(20, round(_real_rate * 100)))
        st.markdown(
            f'<p style="font-size:0.72rem;color:#34d399;font-weight:600;margin:0 0 0.3rem;">'
            f'✅ Using real data from {_n_shifts} logged shifts</p>',
            unsafe_allow_html=True
        )
    else:
        _default_pct = 6
        if _n_shifts > 0:
            st.markdown(
                f'<p style="font-size:0.72rem;color:#94a3b8;margin:0 0 0.3rem;">'
                f'📋 {_n_shifts}/3 shifts logged — keep going to unlock real rate</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p style="font-size:0.72rem;color:#64748b;margin:0 0 0.3rem;">'
                '📋 Log shifts to auto-calculate this</p>',
                unsafe_allow_html=True
            )

    sb_spoilage_pct = st.slider(
        "Area Spoilage Rate", min_value=1, max_value=20,
        value=_default_pct, step=1, format="%d%%",
        help="Auto-filled from logged shift data once 3+ shifts recorded. You can still override manually."
    )
    sb_spoilage_rate = sb_spoilage_pct / 100.0

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">🛒 Cart Status</div>', unsafe_allow_html=True)

    sb_cart_stock = st.number_input(
        "Current Cart Stock", min_value=0, max_value=10_000, value=500, step=10,
        help=f"Carts can be reordered {LEAD_CARTS} day in advance."
    )

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">⚡ Deadline Status</div>', unsafe_allow_html=True)

    r_dl = get_risk("deadline", sb_days_deadline)
    if r_dl == "RED":
        st.error(f"🔴 **{sb_days_deadline} day(s) left — Order now!**")
    elif r_dl == "YELLOW":
        st.warning(f"🟡 **{sb_days_deadline} days — Finalise order soon**")
    else:
        st.success(f"🟢 **{sb_days_deadline} days — On track**")

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.caption(
        f"📦 {PPP} pkgs/pallet · 🛒 {PPC} pkgs/cart\n\n"
        f"Pallet lead: **{LEAD_PALLETS} days** · Cart lead: **{LEAD_CARTS} day**"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER  (dark banner — stays dark, good contrast on light page)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="page-header">
  <div class="header-logo">
    <img src="{get_logo_src()}" style="width:70px;height:70px;object-fit:contain;">
  </div>
  <div class="header-text">
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
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📦  Shift Planner",
    "📅  Weekly Forecast",
    "📋  Shift Log",
    "ℹ️  How It Works",
])


# ┌─────────────────────────────────────────────────────────────────────────────
# │  TAB 1 — SHIFT PLANNER
# └─────────────────────────────────────────────────────────────────────────────
with tab1:

    in_col, res_col = st.columns([0.95, 1.05], gap="large")

    # ── Inputs ──────────────────────────────────────────────────────────────
    with in_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">⚙ Shift Configuration</div>', unsafe_allow_html=True)

        target_volume = st.number_input(
            "📦 Target Volume (packages)",
            min_value=500, max_value=100_000, value=18_000, step=500,
            help="Total packages expected to be processed this shift."
        )

        # Live formula preview
        _lp = formula_pallets(target_volume)
        st.markdown(
            f'<div class="formula-preview">Formula: ⌈ {target_volume:,} ÷ {PPP} ⌉ '
            f'= <span class="formula-result">{_lp} base pallets</span></div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            shift_type = st.selectbox(
                "Shift Type",
                options=["morning","afternoon","night","peak"],
                format_func=lambda x: {"morning":"🌅 Morning","afternoon":"☀️ Afternoon",
                                        "night":"🌙 Night","peak":"⚡ Peak"}[x],
            )
        with c2:
            dow_label   = st.selectbox("Day of Week", options=DAYS_OF_WEEK)
            day_of_week = DAYS_OF_WEEK.index(dow_label)

        c3, c4 = st.columns(2)
        with c3:
            team_size     = st.number_input("Team Size", min_value=5, max_value=150, value=35)
        with c4:
            inbound_ratio = st.slider("Inbound %", 0, 100, 55, 5, format="%d%%") / 100.0

        c5, c6 = st.columns(2)
        with c5:
            is_peak_season = st.checkbox("⚡ Peak Season",
                help="Prime Day, Black Friday, Christmas, etc.")
        with c6:
            has_oversized  = st.checkbox("📏 Oversized Items",
                help="Reduces pallet packing efficiency by 8–15%.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🚀  Run Prediction", key="shift_predict")

    # ── Results ─────────────────────────────────────────────────────────────
    with res_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">📊 Planning Output</div>', unsafe_allow_html=True)

        if predict_clicked:

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
            r_overall  = worst_risk([r_stock, r_deadline, r_spoilage])

            # ── Overall pill + hero ───────────────────────────────────────
            st.markdown(risk_pill(r_overall), unsafe_allow_html=True)
            st.markdown(f"""
            <div class="hero">
              <div class="hero-label">Total Pallets to Order</div>
              <div>
                <span class="hero-num">{total_order}</span>
                <span class="hero-unit">pallets</span>
              </div>
              <div class="hero-chips">
                <span class="hero-chip">📐 Base: {base}</span>
                <span class="hero-chip">🛡 Spoilage: +{buf}</span>
                <span class="hero-chip">🤖 AI adj: {ai_adj:+d}</span>
                <span class="hero-chip">📦 {target_volume:,} ÷ {PPP}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 4 metric chips ────────────────────────────────────────────
            st.markdown(f"""
            <div class="metric-row">
              {metric_chip("Base Pallets",    base,         "#2563eb")}
              {metric_chip("Spoilage Buffer", f"+{buf}",    "#d97706")}
              {metric_chip("Total to Order",  total_order,  "#16a34a")}
              {metric_chip("Carts Needed",    carts_needed, "#7c3aed")}
            </div>
            """, unsafe_allow_html=True)

            # ── Shortage alerts ───────────────────────────────────────────
            if pallet_deficit > 0:
                st.error(
                    f"⚠️ **Pallet shortage — {pallet_deficit} short.** "
                    f"Stock ({sb_pallet_stock}) can't cover demand ({total_order}). Escalate immediately."
                )
            if cart_deficit > 0:
                st.warning(
                    f"🛒 **Cart shortage — {cart_deficit} short.** "
                    f"Order {cart_deficit} extra carts before tomorrow's shift."
                )

            # ── 3 risk cards ──────────────────────────────────────────────
            st.markdown('<div class="section-label">⚡ Risk Breakdown</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="risk-grid">
              {risk_card(r_stock,
                "🟢 Stock — OK" if r_stock=="GREEN" else "🟡 Stock — Monitor" if r_stock=="YELLOW" else "🔴 Stock — Critical",
                f"Stock ({sb_pallet_stock}) covers demand ({total_order}) with headroom." if r_stock=="GREEN"
                else f"{pallet_ratio*100:.0f}% of stock consumed. Confirm next order." if r_stock=="YELLOW"
                else f"Demand ({total_order}) depletes stock ({sb_pallet_stock}). {'Short by '+str(pallet_deficit)+'.' if pallet_deficit else 'Order today.'}"
              )}
              {risk_card(r_deadline,
                "🟢 Deadline — On Track" if r_deadline=="GREEN" else "🟡 Deadline — Plan Now" if r_deadline=="YELLOW" else "🔴 Deadline — Urgent!",
                f"{sb_days_deadline} days to close. Plenty of time." if r_deadline=="GREEN"
                else f"{sb_days_deadline} days left. Finalise your order this week." if r_deadline=="YELLOW"
                else f"Only {sb_days_deadline} day(s) left. Place order TODAY."
              )}
              {risk_card(r_spoilage,
                "🟢 Spoilage — Healthy" if r_spoilage=="GREEN" else "🟡 Spoilage — Elevated" if r_spoilage=="YELLOW" else "🔴 Spoilage — Critical",
                f"{sb_spoilage_pct}% rate is under control. Buffer of +{buf} is adequate." if r_spoilage=="GREEN"
                else f"{sb_spoilage_pct}% is above the 7% target. Review handling procedures." if r_spoilage=="YELLOW"
                else f"{sb_spoilage_pct}% is critically high. Immediate review required."
              )}
            </div>
            """, unsafe_allow_html=True)

            # ── Scenario summary (expander) ───────────────────────────────
            with st.expander("📋  Scenario summary", expanded=False):
                st.dataframe(pd.DataFrame({
                    "Input":  ["Target Volume","Shift Type","Day","Peak Season",
                               "Oversized","Team Size","Inbound Ratio",
                               "Spoilage Rate","Pallet Stock","Days to Deadline"],
                    "Value":  [f"{target_volume:,} pkgs", shift_type.capitalize(), dow_label,
                               "Yes" if is_peak_season else "No",
                               "Yes" if has_oversized  else "No",
                               team_size, f"{inbound_ratio*100:.0f}%",
                               f"{sb_spoilage_pct}%", sb_pallet_stock,
                               f"{sb_days_deadline} day(s)"],
                }), use_container_width=True, hide_index=True)

        else:
            # ── Empty state with live estimate ────────────────────────────
            _q = formula_pallets(target_volume)
            _b = spoilage_buffer(_q, sb_spoilage_rate, "morning", False)
            st.markdown(f"""
            <div class="empty">
              <div class="empty-icon">📦</div>
              <div class="empty-title">Ready to predict</div>
              <div class="empty-body">
                Configure your shift on the left, then click <strong>Run Prediction</strong>.
              </div>
            </div>
            <p style="text-align:center;font-size:0.82rem;color:#94a3b8;margin-top:-0.5rem;">
              Live estimate &nbsp;·&nbsp;
              <b style="color:#2563eb">{_q}</b> base
              + <b style="color:#d97706">{_b}</b> spoilage
              = <b style="color:#16a34a">~{_q+_b} pallets</b>
            </p>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────────────────────────
# │  TAB 2 — WEEKLY FORECAST
# └─────────────────────────────────────────────────────────────────────────────
with tab2:

    st.markdown(
        f"Enter expected daily volumes for the week. MomentumAI calculates your "
        f"**total pallet order** (must be placed **{LEAD_PALLETS} days in advance**) "
        f"and daily cart requirements (adjustable up to **{LEAD_CARTS} day before**)."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Daily volume inputs ──────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">📅 Daily Volume Targets (packages)</div>', unsafe_allow_html=True)

    day_vols     = {}
    default_vols = [35_000, 38_000, 36_000, 40_000, 42_000, 28_000, 20_000]
    d_cols       = st.columns(7)
    for i, (col, day) in enumerate(zip(d_cols, DAYS_OF_WEEK)):
        with col:
            day_vols[day] = col.number_input(
                day[:3], min_value=0, max_value=150_000,
                value=default_vols[i], step=1_000, key=f"wv_{i}"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    wc1, wc2, wc3, wc4 = st.columns(4)
    with wc1:
        weekly_shift = st.selectbox("Predominant Shift", ["morning","afternoon","night","peak"],
                                    format_func=lambda x: x.capitalize(), key="ws")
    with wc2:
        weekly_peak  = st.checkbox("Peak Season Week?", key="wp")
    with wc3:
        weekly_team  = st.number_input("Avg Team Size", 5, 150, 35, key="wt")
    with wc4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_weekly   = st.button("📅  Calculate Weekly Plan", key="wb")

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
            rows.append({"Day": day, "Volume": f"{vol:,}",
                         "Base Pallets": base, "Buffer": buf,
                         "Total Pallets": order, "Carts": carts})
            total_pallet_order += order

        weekly_df      = pd.DataFrame(rows)
        total_carts_wk = sum(formula_carts(day_vols[d]) for d in DAYS_OF_WEEK)
        total_vol_wk   = sum(day_vols.values())

        r_wk = worst_risk([
            get_risk("stock",    total_pallet_order / max(sb_pallet_stock, 1)),
            get_risk("deadline", sb_days_deadline),
            get_risk("spoilage", sb_spoilage_rate),
        ])

        # ── Summary metrics ───────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
          {metric_chip("Weekly Pallets",  total_pallet_order,  "#16a34a")}
          {metric_chip("Weekly Carts",    total_carts_wk,      "#7c3aed")}
          {metric_chip("Total Volume",    f"{total_vol_wk:,}", "#2563eb")}
          {metric_chip("Order Deadline",  f"{sb_days_deadline}d", "#d97706")}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(risk_pill(r_wk), unsafe_allow_html=True)

        if get_risk("deadline", sb_days_deadline) == "RED":
            st.error(f"🔴 **Order today!** {sb_days_deadline} day(s) left — place order for **{total_pallet_order} pallets** immediately.")
        elif get_risk("deadline", sb_days_deadline) == "YELLOW":
            st.warning(f"🟡 **{sb_days_deadline} days to deadline.** Finalise order for **{total_pallet_order} pallets** this week.")
        else:
            st.success(f"🟢 On track. Order **{total_pallet_order} pallets** before the {LEAD_PALLETS}-day deadline.")

        ra, rb = st.columns([1.1, 0.9], gap="large")
        with ra:
            st.markdown('<div class="section-label">📊 Daily Breakdown</div>', unsafe_allow_html=True)
            st.dataframe(weekly_df, use_container_width=True, hide_index=True)
        with rb:
            st.markdown('<div class="section-label">📈 Volume by Day</div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({
                "Day":    [d[:3] for d in DAYS_OF_WEEK],
                "Volume": [day_vols[d] for d in DAYS_OF_WEEK],
            }).set_index("Day")
            st.bar_chart(chart_df, color="#2563eb", use_container_width=True)

    else:
        st.markdown("""
        <div class="empty">
          <div class="empty-icon">📅</div>
          <div class="empty-title">Adjust daily volumes above</div>
          <div class="empty-body">Then click <strong>Calculate Weekly Plan</strong>.</div>
        </div>
        """, unsafe_allow_html=True)


# ┌─────────────────────────────────────────────────────────────────────────────
# │  TAB 3 — SHIFT LOG
# └─────────────────────────────────────────────────────────────────────────────
with tab3:

    # ── Summary KPI chips ────────────────────────────────────────────────────
    stats = get_summary_stats()

    st.markdown("<br>", unsafe_allow_html=True)

    if stats["total_shifts"] == 0:
        st.markdown("""
        <div class="empty">
          <div class="empty-icon">📋</div>
          <div class="empty-title">No shifts logged yet</div>
          <div class="empty-body">
            Use the form below to log your first completed shift.
            After 3+ shifts the app will automatically use your real spoilage data
            instead of the manual slider.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        spoilage_display = (
            f"{stats['spoilage_rate']*100:.1f}%"
            if stats["spoilage_rate"] is not None else "—"
        )
        accuracy_display = (
            f"{stats['avg_accuracy_pct']}%"
            if stats["avg_accuracy_pct"] is not None else "—"
        )
        st.markdown(f"""
        <div class="metric-row">
          {metric_chip("Shifts Logged",   stats["total_shifts"],          "#2563eb")}
          {metric_chip("Total Volume",    f"{stats['total_volume']:,}",   "#7c3aed")}
          {metric_chip("Pallets Ordered", stats["total_ordered"],         "#16a34a")}
          {metric_chip("Pallets Spoilt",  stats["total_spoilt"],          "#d97706")}
          {metric_chip("Real Spoilage",   spoilage_display,               "#dc2626")}
          {metric_chip("Pred Accuracy",   accuracy_display,               "#0891b2")}
        </div>
        """, unsafe_allow_html=True)

        real_rate, n_shifts = get_real_spoilage_rate()
        if real_rate is not None:
            st.success(
                f"✅ **Real spoilage rate active** — calculated from {n_shifts} logged shifts "
                f"({real_rate*100:.1f}%). The sidebar slider has been replaced with this live figure."
            )
        else:
            remaining = max(0, 3 - n_shifts)
            st.info(
                f"ℹ️ Log **{remaining} more shift(s)** to unlock automatic spoilage tracking. "
                f"The sidebar slider is still active."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Log a shift form ─────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">📝 Log a Completed Shift</div>', unsafe_allow_html=True)

    with st.form("log_shift_form", clear_on_submit=True):
        lf1, lf2, lf3 = st.columns(3)
        with lf1:
            log_date       = st.date_input("Shift Date", value=date.today())
            log_shift_type = st.selectbox(
                "Shift Type", ["morning", "afternoon", "night", "peak"],
                format_func=lambda x: x.capitalize()
            )
        with lf2:
            log_volume     = st.number_input("Target Volume (pkgs)", 1_000, 150_000, 35_000, 1_000)
            log_ordered    = st.number_input("Pallets Ordered", 1, 500, 20)
        with lf3:
            log_spoilt     = st.number_input("Pallets Spoilt / Damaged", 0, 100, 1)
            log_carts      = st.number_input("Carts Ordered", 0, 500, 10)

        lf4, lf5 = st.columns(2)
        with lf4:
            log_peak_season = st.checkbox("Peak Season Shift?")
            log_oversized   = st.checkbox("Had Oversized Items?")
        with lf5:
            log_notes = st.text_area("Notes (optional)", placeholder="e.g. delayed inbound, understaffed…", height=80)

        submitted = st.form_submit_button("💾  Save Shift", use_container_width=True)

    if submitted:
        # Work out what the app would have predicted for this shift
        _base_pred  = formula_pallets(log_volume)
        _buf_pred   = spoilage_buffer(_base_pred, sb_spoilage_rate, log_shift_type, log_peak_season)
        _pallet_pred = _base_pred + _buf_pred
        _cart_pred   = formula_carts(log_volume)

        log_shift(
            shift_date        = log_date,
            day_of_week       = log_date.strftime("%A"),
            shift_type        = log_shift_type,
            target_volume     = log_volume,
            pallets_predicted = _pallet_pred,
            carts_predicted   = _cart_pred,
            pallets_ordered   = log_ordered,
            pallets_spoilt    = log_spoilt,
            carts_ordered     = log_carts,
            is_peak_season    = log_peak_season,
            has_oversized     = log_oversized,
            notes             = log_notes,
        )
        st.success(f"✅ Shift logged for {log_date.strftime('%A %d %b %Y')}. Keep building your history!")
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── History table ─────────────────────────────────────────────────────────
    history_df = get_history()

    if not history_df.empty:
        st.markdown('<div class="section-label">🗂️ Shift History</div>', unsafe_allow_html=True)

        # Friendly display columns
        display_df = history_df[[
            "id", "shift_date", "day_of_week", "shift_type",
            "target_volume", "pallets_predicted", "pallets_ordered",
            "pallets_spoilt", "carts_ordered", "notes",
        ]].copy()
        display_df.columns = [
            "ID", "Date", "Day", "Shift",
            "Volume", "Predicted Pallets", "Ordered Pallets",
            "Spoilt", "Carts Ordered", "Notes",
        ]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Delete a row ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🗑️  Delete a row (correct a mistake)"):
            del_id = st.number_input(
                "Enter the ID of the row to delete",
                min_value=1, step=1,
                help="The ID is shown in the leftmost column of the table above."
            )
            if st.button("Delete Row", type="secondary"):
                if int(del_id) in history_df["id"].values:
                    delete_shift(int(del_id))
                    st.success(f"Row {int(del_id)} deleted.")
                    st.rerun()
                else:
                    st.error(f"No row found with ID {int(del_id)}.")


# ┌─────────────────────────────────────────────────────────────────────────────
# │  TAB 4 — HOW IT WORKS
# └─────────────────────────────────────────────────────────────────────────────
with tab4:

    hw1, hw2 = st.columns(2, gap="large")

    with hw1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### 📐 The Core Formula")
        st.markdown(f"""
Every prediction starts with the fundamental warehouse math:

> **Base Pallets = ⌈ Target Volume ÷ {PPP} ⌉**

A shift processing **40,000 packages** requires:

> ⌈ 40,000 ÷ {PPP} ⌉ = **{math.ceil(40000/PPP)} pallets**

The AI model refines this using learned patterns — shift type, seasonality, team workload, and your area's spoilage history.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### 🛡️ Spoilage Buffer System")
        st.markdown(f"""
Damaged/unusable pallets are unavoidable. The buffer compounds three factors:

| Factor | Multiplier |
|--------|-----------|
| Area spoilage rate | Your input (default {CONFIG['base_spoilage_rate']*100:.0f}%) |
| Night shift | ×{CONFIG['spoilage_by_shift']['night']:.2f} (reduced supervision) |
| Peak shift | ×{CONFIG['spoilage_by_shift']['peak']:.2f} (rushed pace) |
| Peak season | ×{CONFIG['peak_season_spoilage_multiplier']:.2f} extra |
| Safety floor | Always +{CONFIG['min_extra_pallets']} pallets minimum |

> Buffer = max({CONFIG['min_extra_pallets']}, ⌈ Base × Rate × Shift × Season ⌉)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with hw2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### 🔴🟡🟢 Risk Alert System")
        st.markdown(f"""
Three independent dimensions evaluated in parallel:

**1. Pallet Stock** — demand vs. on-hand
- 🟢 Need < {RISK_CFG['stock_green']*100:.0f}% of stock
- 🟡 Need {RISK_CFG['stock_green']*100:.0f}–{RISK_CFG['stock_yellow']*100:.0f}% of stock
- 🔴 Need > {RISK_CFG['stock_yellow']*100:.0f}%

**2. Order Deadline** — days until weekly close
- 🟢 > {RISK_CFG['deadline_green']} days
- 🟡 {RISK_CFG['deadline_red']+1}–{RISK_CFG['deadline_green']} days — finalise now
- 🔴 ≤ {RISK_CFG['deadline_red']} days — order today!

**3. Spoilage Rate** — is damage under control?
- 🟢 < {RISK_CFG['spoilage_green']*100:.0f}%
- 🟡 {RISK_CFG['spoilage_green']*100:.0f}–{RISK_CFG['spoilage_yellow']*100:.0f}% — review handling
- 🔴 > {RISK_CFG['spoilage_yellow']*100:.0f}% — immediate action

Overall risk = worst of the three.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### 🛒 Pallets vs Carts")
        st.markdown(f"""
| | Pallets | Carts |
|-|---------|-------|
| **Capacity** | {PPP} pkgs | {PPC} pkgs |
| **Order lead time** | **{LEAD_PALLETS} days** | **{LEAD_CARTS} day** |
| **If you miss it** | Cannot recover | Usually fixable same day |

Pallets are the primary problem — the **{LEAD_PALLETS}-day lead time** leaves no room for error.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Reference bar
    st.markdown(f"""
    <div class="ref-bar">
      <div><div class="ref-label">Pallet Capacity</div><div class="ref-value">{PPP} pkgs</div></div>
      <div><div class="ref-label">Cart Capacity</div><div class="ref-value">{PPC} pkgs</div></div>
      <div><div class="ref-label">Pallet Lead Time</div><div class="ref-value">{LEAD_PALLETS} days</div></div>
      <div><div class="ref-label">Cart Lead Time</div><div class="ref-value">{LEAD_CARTS} day</div></div>
      <div><div class="ref-label">Default Spoilage</div><div class="ref-value">{CONFIG['base_spoilage_rate']*100:.0f}%</div></div>
      <div><div class="ref-label">Min Buffer</div><div class="ref-value">+{CONFIG['min_extra_pallets']} pallets</div></div>
    </div>
    """, unsafe_allow_html=True)
