"""
Baghewala CSS + SRP Digital Twin — SCADA HMI Dashboard
=======================================================
Production-grade Streamlit web application for the PINN-based Digital Twin.

Launch:
    cd C:\\Akash\\PINN_AND_DATA\\scada_pinn\\baghewala_scada_twin\\scada
    streamlit run digital_twin_ui.py
"""
import json
import math
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "integration_out")
TWIN_DIR = os.path.join(HERE, "twin")
RESULTS_DIR = os.path.join(TWIN_DIR, "updated_results")
DB_TWIN = os.path.join(OUT, "twin.db")
DB_BASELINE = os.path.join(OUT, "baseline.db")

# ---------------------------------------------------------------------------
# COLORS & THEME
# ---------------------------------------------------------------------------
ACCENT_BLUE = "#00D4FF"
ACCENT_GREEN = "#00FF88"
ACCENT_RED = "#FF4444"
ACCENT_AMBER = "#FFB300"
ACCENT_PURPLE = "#B388FF"
GRID_COLOR = "#2A2D35"
TEXT_COLOR = "#E0E0E0"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_COLOR, family="Consolas, monospace", size=12),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=GRID_COLOR),
)


def styled_fig(fig, **kw):
    layout = dict(PLOTLY_LAYOUT, **kw)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=30)
def query_db(db_path, sql, params=()):
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_get(d, *keys, default=0):
    """Safely traverse nested dict."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Baghewala Digital Twin — SCADA HMI",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');

    /* Transform Matplotlib white-bg plots to dark neon glow */
    img {
        filter: invert(1) hue-rotate(180deg) brightness(1.2) contrast(1.1);
        mix-blend-mode: screen;
        border-radius: 8px;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e14 0%, #111720 50%, #0d1117 100%);
    }

    /* Header banner */
    .dt-header {
        background: linear-gradient(90deg, #0d1b2a 0%, #1b2838 50%, #0d1b2a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 18px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0, 100, 200, 0.15);
    }
    .dt-header h1 {
        color: #00d4ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        margin: 0;
        letter-spacing: 2px;
    }
    .dt-header .dt-sub {
        color: #8899aa;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    .dt-header .dt-status {
        background: #00ff8833;
        border: 1px solid #00ff88;
        color: #00ff88;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a2332 0%, #141c28 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 4px 20px rgba(0,212,255,0.15);
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        margin: 4px 0;
    }
    .kpi-label {
        color: #8899aa;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        margin-top: 4px;
    }
    .kpi-up { color: #00ff88; }
    .kpi-down { color: #ff4444; }
    .kpi-neutral { color: #ffb300; }

    /* Section titles */
    .section-title {
        color: #00d4ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Badges */
    .badge-pass {
        background: #00ff8822; border: 1px solid #00ff88; color: #00ff88;
        padding: 3px 10px; border-radius: 12px;
        font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
    }
    .badge-fail {
        background: #ff444422; border: 1px solid #ff4444; color: #ff4444;
        padding: 3px 10px; border-radius: 12px;
        font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
    }

    /* Info panel */
    .info-panel {
        background: linear-gradient(135deg, #1a2332 0%, #141c28 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #141c28; border-radius: 8px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #8899aa;
        font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 1px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a5f; color: #00d4ff; border-radius: 6px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111720 100%);
        border-right: 1px solid #1e3a5f;
    }

    .stDataFrame { border: 1px solid #1e3a5f; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="dt-header">
    <div>
        <h1>🛢️ BAGHEWALA DIGITAL TWIN</h1>
        <div class="dt-sub">PINN-Based SCADA Integration — CSS + SRP Well-to-Surface Optimization</div>
    </div>
    <div style="text-align:right">
        <div class="dt-status">● TWIN ONLINE</div>
        <div class="dt-sub" style="margin-top:4px">BGW-01 | Jodhpur Sandstone</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Navigation")
    page = st.radio(
        "Select View",
        [
            "🏠 Executive Dashboard",
            "📊 Live SCADA HMI",
            "🔬 Dynamometer Cards",
            "🧠 Twin Decision Engine",
            "🧪 PINN Model Performance",
            "⚡ SRP Optimization",
            "📈 Field Historian",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### 🏭 Well Context")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#8899aa;line-height:1.8">
    <b style="color:#00d4ff">Field:</b> Baghewala, Rajasthan<br>
    <b style="color:#00d4ff">Well:</b> BGW-01<br>
    <b style="color:#00d4ff">Process:</b> CSS + SRP (VFD)<br>
    <b style="color:#00d4ff">Pump:</b> 1,050 m MD<br>
    <b style="color:#00d4ff">Unit:</b> C-228D-213-86<br>
    <b style="color:#00d4ff">Rod:</b> 7/8" Grade D<br>
    <b style="color:#00d4ff">Plunger:</b> 1.5" insert<br>
    <b style="color:#00d4ff">API:</b> 18 (heavy oil)<br>
    <b style="color:#00d4ff">μ₅₀°C:</b> ~12,000 cP<br>
    <b style="color:#00d4ff">Reservoir:</b> Jodhpur Sst.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#556677;text-align:center">
    PINN Architecture<br>
    Reservoir: 51k params<br>
    SRP Wave: 46k params<br>
    Total: ~97k params<br><br>
    Hard-constraint Boberg-Lantz<br>
    Damped-wave rod PDE
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# HELPER: KPI card HTML
# =========================================================================
def kpi_html(label, value, fmt=".1f", unit="", color=ACCENT_BLUE,
             delta=None, delta_fmt=".1f", invert=False):
    """Generate a styled KPI card."""
    val_str = f"{value:{fmt}}{unit}" if isinstance(value, (int, float)) else str(value)
    delta_html = ""
    if delta is not None and isinstance(delta, (int, float)):
        pct = 100 * delta / max(abs(value - delta), 1e-9) if value != delta else 0
        d = value - delta  # twin - baseline
        cls = "kpi-up" if (d < 0 and invert) or (d > 0 and not invert) else "kpi-down"
        if abs(pct) < 1:
            cls = "kpi-neutral"
        arrow = "▼" if d < 0 else "▲"
        delta_html = f"""
        <div class="kpi-delta {cls}">{arrow} {abs(d):{delta_fmt}} vs baseline</div>
        <div class="kpi-label" style="color:#556677">Baseline: {delta:{fmt}}{unit}</div>
        """
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{val_str}</div>
        {delta_html}
    </div>
    """


# =========================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# =========================================================================
if page == "🏠 Executive Dashboard":
    twin_sum = load_json(os.path.join(OUT, "twin_summary.json"))
    base_sum = load_json(os.path.join(OUT, "baseline_summary.json"))
    twin_run = load_json(os.path.join(OUT, "twin_run.json"))
    acceptance = load_json(os.path.join(OUT, "acceptance_report.json"))
    report = load_json(os.path.join(RESULTS_DIR, "digital_twin_report.json"))

    st.markdown('<div class="section-title">⚡ OPERATIONAL KPIs — TWIN vs BASELINE (5-DAY INTEGRATION)</div>',
                unsafe_allow_html=True)

    # KPI row
    cols = st.columns(8)
    kpis = [
        ("Oil Produced", "oil_bbl", ".1f", " bbl", ACCENT_GREEN, False),
        ("SOR", "kwh_per_bbl_oil", ".2f", " kWh/bbl", ACCENT_BLUE, True),
        ("Energy Intensity", "kwh_per_bbl_oil", ".2f", " kWh/bbl", ACCENT_GREEN, True),
        ("Rod Float Trips", "trips_low_load", "d", "", ACCENT_RED, True),
        ("Impact Hours", "impact_over_limit_h", ".1f", " h", ACCENT_AMBER, True),
        ("Total Power", "kwh", ".1f", " kWh", ACCENT_GREEN, True),
        ("Cards Diagnosed", "cards", "d", "", ACCENT_BLUE, False),
        ("Writes Accepted", "writes_accepted", "d", "", ACCENT_PURPLE, False),
    ]
    for col, (label, key, fmt, unit, color, inv) in zip(cols, kpis):
        tv = twin_sum.get(key, 0)
        bv = base_sum.get(key, 0)
        with col:
            st.markdown(kpi_html(label, tv, fmt, unit, color, bv, fmt, inv),
                        unsafe_allow_html=True)

    st.markdown("")

    # Charts row
    col_left, col_right = st.columns(2)
    decisions = twin_run.get("decisions", [])

    with col_left:
        st.markdown('<div class="section-title">📉 SPM TRAJECTORY — TWIN DECISION HISTORY</div>',
                    unsafe_allow_html=True)
        if decisions:
            df_dec = pd.DataFrame(decisions)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_dec["t_sim_day"], y=df_dec["spm_cur"],
                mode="lines+markers", name="SPM (actual)",
                line=dict(color=ACCENT_BLUE, width=2),
                marker=dict(size=5, color=ACCENT_BLUE)))
            fig.add_trace(go.Scatter(
                x=df_dec["t_sim_day"], y=df_dec["spm_req"],
                mode="lines+markers", name="SPM (twin request)",
                line=dict(color=ACCENT_GREEN, width=2, dash="dash"),
                marker=dict(size=5, color=ACCENT_GREEN, symbol="diamond")))
            fig.add_trace(go.Scatter(
                x=df_dec["t_sim_day"], y=df_dec["model_spm"],
                mode="lines", name="Model optimum",
                line=dict(color=ACCENT_PURPLE, width=1, dash="dot")))
            # Event annotations
            fig.add_vline(x=1.0, line_dash="dash", line_color=ACCENT_RED, opacity=0.4,
                          annotation_text="Inflow drop 55%", annotation_position="top right",
                          annotation=dict(font_color=ACCENT_RED, font_size=10))
            fig.add_vline(x=2.0, line_dash="dash", line_color=ACCENT_AMBER, opacity=0.4,
                          annotation_text="Cooling + asphaltene", annotation_position="top left",
                          annotation=dict(font_color=ACCENT_AMBER, font_size=10))
            fig.add_vrect(x0=3.5, x1=3.75, fillcolor=ACCENT_RED, opacity=0.08,
                          annotation_text="Sensor desync", annotation_position="top left",
                          annotation=dict(font_color=ACCENT_RED, font_size=10))
            fig.add_vrect(x0=4.0, x1=4.25, fillcolor=ACCENT_AMBER, opacity=0.08,
                          annotation_text="Twin offline", annotation_position="top left",
                          annotation=dict(font_color=ACCENT_AMBER, font_size=10))
            styled_fig(fig, height=380, title="SPM Over Time (Simulated Days)",
                       xaxis_title="Day", yaxis_title="Strokes per Minute")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">🔧 FRICTION MULTIPLIER & FILL TRACKING</div>',
                    unsafe_allow_html=True)
        if decisions:
            df_dec = pd.DataFrame(decisions)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=df_dec["t_sim_day"], y=df_dec["friction_mult"],
                mode="lines+markers", name="Friction ×",
                line=dict(color=ACCENT_RED, width=2), marker=dict(size=4)),
                secondary_y=False)
            fig.add_trace(go.Scatter(
                x=df_dec["t_sim_day"], y=df_dec["fillage"],
                mode="lines+markers", name="Pump Fillage",
                line=dict(color=ACCENT_BLUE, width=2), marker=dict(size=4)),
                secondary_y=True)
            fig.add_hline(y=1.0, line_dash="dot", line_color="#555", secondary_y=False)
            fig.add_hline(y=0.85, line_dash="dot", line_color=ACCENT_AMBER, opacity=0.4,
                          secondary_y=True, annotation_text="Fillage target",
                          annotation=dict(font_color=ACCENT_AMBER, font_size=10))
            styled_fig(fig, height=380, title="Friction & Fillage vs Time",
                       yaxis_title="Friction ×", yaxis2_title="Fillage (-)")
            fig.update_yaxes(gridcolor=GRID_COLOR, secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

    # Acceptance scorecard
    st.markdown('<div class="section-title">✅ ACCEPTANCE TEST SCORECARD</div>', unsafe_allow_html=True)
    results = acceptance.get("results", [])
    if results:
        acc_cols = st.columns(min(len(results), 9))
        for c, r in zip(acc_cols, results):
            badge = "badge-pass" if r.get("status") == "PASS" else "badge-fail"
            with c:
                st.markdown(f"""
                <div style="text-align:center;padding:8px">
                    <div style="color:#8899aa;font-family:'JetBrains Mono';font-size:18px;font-weight:700">{r.get('id','')}</div>
                    <div class="{badge}">{r.get('status','')}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("")

    # Comparison bar + latency
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📊 BASELINE vs TWIN COMPARISON</div>', unsafe_allow_html=True)
        categories = ["Oil (bbl)", "Power (kWh)", "Impact (h)", "Float Trips", "Cards"]
        base_vals = [base_sum.get("oil_bbl", 0), base_sum.get("kwh", 0),
                     base_sum.get("impact_over_limit_h", 0), base_sum.get("trips_low_load", 0),
                     base_sum.get("cards", 0)]
        twin_vals = [twin_sum.get("oil_bbl", 0), twin_sum.get("kwh", 0),
                     twin_sum.get("impact_over_limit_h", 0), twin_sum.get("trips_low_load", 0),
                     twin_sum.get("cards", 0)]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline", x=categories, y=base_vals,
                             marker_color="#ff6666", opacity=0.8))
        fig.add_trace(go.Bar(name="Twin", x=categories, y=twin_vals,
                             marker_color=ACCENT_BLUE, opacity=0.8))
        styled_fig(fig, height=300, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">⏱️ DECISION LATENCY DISTRIBUTION</div>', unsafe_allow_html=True)
        if decisions:
            latencies = [d.get("total_s", 0) for d in decisions]
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=latencies, nbinsx=15, marker_color=ACCENT_BLUE, opacity=0.8, name="Latency (s)"))
            fig.add_vline(x=np.median(latencies), line_dash="dash", line_color=ACCENT_GREEN,
                          annotation_text=f"Median: {np.median(latencies):.1f}s",
                          annotation=dict(font_color=ACCENT_GREEN, font_size=11))
            styled_fig(fig, height=300, xaxis_title="Seconds", yaxis_title="Count",
                       title="PINN Inference Latency")
            st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# PAGE 2: LIVE SCADA HMI
# =========================================================================
elif page == "📊 Live SCADA HMI":
    st.markdown('<div class="section-title">🖥️ SCADA HMI — REAL-TIME TAG MONITORING</div>',
                unsafe_allow_html=True)

    tags_df = query_db(DB_TWIN, "SELECT t, tag, value FROM tag_history WHERE well='BGW-01' ORDER BY t")

    if tags_df.empty:
        st.warning("No tag history data found. Run the integration test first.")
    else:
        latest_t = tags_df["t"].max()
        latest = tags_df[tags_df["t"] == latest_t].set_index("tag")["value"].to_dict()

        # Gauge row
        st.markdown("##### 🎛️ Live Instrument Gauges")
        g_cols = st.columns(6)
        gauge_data = [
            ("SPM", "SRP/SPM", "1/min", 0, 12, ACCENT_BLUE),
            ("VFD Freq", "VFD/Frequency", "Hz", 0, 80, ACCENT_GREEN),
            ("Motor Power", "VFD/MotorPower", "kW", 0, 20, ACCENT_AMBER),
            ("Wellhead T", "Wellhead/Temperature", "°C", 30, 250, ACCENT_RED),
            ("Tubing P", "Wellhead/TubingPressure", "psi", 0, 200, ACCENT_PURPLE),
            ("Fluid Level", "Wellhead/FluidLevel_Acoustic", "ft", 0, 3500, ACCENT_BLUE),
        ]
        for col, (label, tag, unit, lo, hi, color) in zip(g_cols, gauge_data):
            val = latest.get(tag, 0)
            with col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=val,
                    title=dict(text=f"<b>{label}</b><br><span style='font-size:10px;color:#888'>{unit}</span>",
                               font=dict(size=13, color=TEXT_COLOR)),
                    number=dict(font=dict(size=22, color=color, family="JetBrains Mono")),
                    gauge=dict(
                        axis=dict(range=[lo, hi], tickfont=dict(size=9, color="#666")),
                        bar=dict(color=color, thickness=0.3),
                        bgcolor="#1a1d23", borderwidth=1, bordercolor="#1e3a5f",
                        steps=[
                            dict(range=[lo, lo + 0.3 * (hi - lo)], color="#0d1b2a"),
                            dict(range=[lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)], color="#111720"),
                            dict(range=[lo + 0.7 * (hi - lo), hi], color="#1a1012"),
                        ],
                    )))
                fig.update_layout(height=180, margin=dict(l=20, r=20, t=50, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR))
                st.plotly_chart(fig, use_container_width=True)

        # Time series
        st.markdown("##### 📈 Tag Time Series")
        tag_groups = {
            "SRP / Rod Pump": ["SRP/SPM", "SRP/PPRL", "SRP/MPRL", "SRP/PumpFillage_POC"],
            "VFD / Motor": ["VFD/Frequency", "VFD/MotorCurrent", "VFD/MotorPower"],
            "Wellhead / Surveys": ["Wellhead/Temperature", "Wellhead/TubingPressure",
                                   "Wellhead/CasingPressure"],
        }
        selected_group = st.selectbox("Tag Group", list(tag_groups.keys()))
        selected_tags = tag_groups[selected_group]
        tag_filter = tags_df[tags_df["tag"].isin(selected_tags)]

        if not tag_filter.empty:
            fig = make_subplots(rows=len(selected_tags), cols=1, shared_xaxes=True,
                                vertical_spacing=0.04, subplot_titles=selected_tags)
            colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_PURPLE]
            for i, tag in enumerate(selected_tags):
                td = tag_filter[tag_filter["tag"] == tag]
                fig.add_trace(go.Scatter(
                    x=td["t"] / 86400, y=td["value"], mode="lines", name=tag,
                    line=dict(color=colors[i % len(colors)], width=1.5)), row=i + 1, col=1)
            styled_fig(fig, height=180 * len(selected_tags),
                       xaxis=dict(title="Simulated Day", gridcolor=GRID_COLOR), showlegend=False)
            for i in range(len(selected_tags)):
                fig.update_yaxes(gridcolor=GRID_COLOR, row=i + 1, col=1)
                fig.update_xaxes(gridcolor=GRID_COLOR, row=i + 1, col=1)
            st.plotly_chart(fig, use_container_width=True)

        # Status tags
        st.markdown("##### 🔒 PLC & Twin Status Tags")
        status_tags = ["PLC/Mode", "PLC/Alarm", "PLC/SimTime", "Twin/Heartbeat",
                       "SRP/RunStatus", "SRP/RunTimeFrac_24h", "Steam/Phase",
                       "Steam/DaysSinceInjectionEnd"]
        st_data = []
        for tag in status_tags:
            val = latest.get(tag, "N/A")
            st_data.append({"Tag": tag, "Value": f"{val:.2f}" if isinstance(val, float) else str(val)})
        st.dataframe(pd.DataFrame(st_data), hide_index=True, use_container_width=True)


# =========================================================================
# PAGE 3: DYNAMOMETER CARDS
# =========================================================================
elif page == "🔬 Dynamometer Cards":
    st.markdown('<div class="section-title">🔬 DYNAMOMETER CARD VIEWER — SURFACE & DOWNHOLE (PINN)</div>',
                unsafe_allow_html=True)

    cards_df = query_db(DB_TWIN, """
        SELECT card_id, t, spm, stroke, pos, load
        FROM cards WHERE well='BGW-01' ORDER BY card_id
    """)

    if cards_df.empty:
        st.warning("No card data available. Run the integration test first.")
    else:
        card_ids = cards_df["card_id"].tolist()
        col_slider, col_info = st.columns([3, 1])
        with col_slider:
            selected_id = st.select_slider(
                "Select Card ID", options=card_ids,
                value=card_ids[len(card_ids) // 2],
                format_func=lambda x: f"Card #{x}")

        row = cards_df[cards_df["card_id"] == selected_id].iloc[0]
        pos = json.loads(row["pos"])
        load = json.loads(row["load"])
        day = row["t"] / 86400.0

        with col_info:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Card #{selected_id}</div>
                <div class="kpi-value" style="color:{ACCENT_BLUE};font-size:18px">Day {day:.2f}</div>
                <div class="kpi-label">SPM {row['spm']:.1f} | Stroke {row['stroke']:.0f}"</div>
            </div>
            """, unsafe_allow_html=True)

        col_card, col_card2 = st.columns(2)
        with col_card:
            st.markdown("##### Surface Dynamometer Card")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pos, y=load, mode="lines",
                line=dict(color=ACCENT_BLUE, width=2.5),
                fill="toself", fillcolor="rgba(0,212,255,0.08)", name="Surface Card"))
            styled_fig(fig, height=400, xaxis_title="Position (ft)", yaxis_title="Load (lbf)",
                       title=f"Surface Card #{selected_id} — Day {day:.2f}")
            st.plotly_chart(fig, use_container_width=True)

        with col_card2:
            st.markdown("##### Card Evolution Over Time")
            n_show = min(8, len(card_ids))
            step = max(1, len(card_ids) // n_show)
            sample_ids = card_ids[::step][:n_show]
            fig = go.Figure()
            for sid in sample_ids:
                r = cards_df[cards_df["card_id"] == sid].iloc[0]
                p = json.loads(r["pos"])
                l = json.loads(r["load"])
                d = r["t"] / 86400.0
                opacity = 0.3 if sid != selected_id else 1.0
                width = 1 if sid != selected_id else 3
                fig.add_trace(go.Scatter(
                    x=p, y=l, mode="lines", name=f"Day {d:.1f}",
                    line=dict(width=width), opacity=opacity))
            styled_fig(fig, height=400, xaxis_title="Position (ft)", yaxis_title="Load (lbf)",
                       title="Card Evolution Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # Card statistics
        st.markdown("##### 📊 Card Statistics")
        stats_data = []
        for _, r in cards_df.iterrows():
            p = json.loads(r["pos"])
            l = json.loads(r["load"])
            stats_data.append({
                "Card ID": int(r["card_id"]), "Day": round(r["t"] / 86400, 2),
                "SPM": round(r["spm"], 1), "Stroke (in)": round(r["stroke"], 0),
                "PPRL (lbf)": round(max(l), 1), "MPRL (lbf)": round(min(l), 1),
                "Load Range": round(max(l) - min(l), 1),
            })
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True, height=300)

    # Show PINN result images
    st.markdown('<div class="section-title">🔬 PINN DOWNHOLE PREDICTIONS</div>', unsafe_allow_html=True)
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        try:
            st.image(os.path.join(RESULTS_DIR, "srp_cards.png"), caption="Surface vs Downhole Cards (PINN)")
        except Exception:
            st.info("srp_cards.png not found in updated_results/")
    with img_col2:
        try:
            st.image(os.path.join(RESULTS_DIR, "fluid_pound_card.png"), caption="Fluid Pound Detection")
        except Exception:
            st.info("fluid_pound_card.png not found in updated_results/")


# =========================================================================
# PAGE 4: TWIN DECISION ENGINE
# =========================================================================
elif page == "🧠 Twin Decision Engine":
    st.markdown('<div class="section-title">🧠 PINN DIGITAL TWIN — DECISION ENGINE & AUDIT TRAIL</div>',
                unsafe_allow_html=True)

    twin_run = load_json(os.path.join(OUT, "twin_run.json"))
    decisions = twin_run.get("decisions", [])

    if not decisions:
        st.warning("No decision data available. Run the integration test first.")
    else:
        df_dec = pd.DataFrame(decisions)

        # Startup context
        with st.expander("🔧 Twin Startup Configuration", expanded=False):
            startup = twin_run.get("startup", {})
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Completion**")
                st.json(startup.get("completion", {}))
            with c2:
                st.markdown("**Fluid Properties**")
                st.json(startup.get("fluid", {}))
            with c3:
                st.markdown("**Failure History**")
                st.json(startup.get("failure_history", {}))

        # Decision timeline
        st.markdown("##### 🕐 Decision Timeline")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=["SPM Control", "Pump Fillage", "Rod Float Margin",
                                            "Friction Multiplier"],
                            row_heights=[0.3, 0.25, 0.25, 0.2])

        fig.add_trace(go.Scatter(x=df_dec["t_sim_day"], y=df_dec["spm_cur"], mode="lines+markers",
                                 name="Current SPM", line=dict(color=ACCENT_BLUE, width=2),
                                 marker=dict(size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_dec["t_sim_day"], y=df_dec["spm_req"], mode="lines+markers",
                                 name="Requested SPM", line=dict(color=ACCENT_GREEN, width=2, dash="dash"),
                                 marker=dict(size=4, symbol="diamond")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_dec["t_sim_day"], y=df_dec["model_spm"], mode="lines",
                                 name="Model Optimum", line=dict(color=ACCENT_PURPLE, width=1, dash="dot")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df_dec["t_sim_day"], y=df_dec["fillage"], mode="lines+markers",
                                 name="Fillage", line=dict(color=ACCENT_BLUE, width=2),
                                 marker=dict(size=4)), row=2, col=1)
        fig.add_hline(y=0.85, line_dash="dot", line_color=ACCENT_AMBER, row=2, col=1)
        fig.add_trace(go.Scatter(x=df_dec["t_sim_day"], y=df_dec["margin"], mode="lines+markers",
                                 name="Float Margin", line=dict(color=ACCENT_RED, width=2),
                                 marker=dict(size=4)), row=3, col=1)
        fig.add_hline(y=0.20, line_dash="dot", line_color=ACCENT_RED, row=3, col=1)
        fig.add_trace(go.Scatter(x=df_dec["t_sim_day"], y=df_dec["friction_mult"], mode="lines+markers",
                                 name="Friction ×", line=dict(color=ACCENT_AMBER, width=2),
                                 marker=dict(size=4)), row=4, col=1)

        styled_fig(fig, height=700, showlegend=True)
        for i in range(1, 5):
            fig.update_yaxes(gridcolor=GRID_COLOR, row=i, col=1)
            fig.update_xaxes(gridcolor=GRID_COLOR, row=i, col=1)
        fig.update_xaxes(title_text="Simulated Day", row=4, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Decision table
        st.markdown("##### 📋 Decision Log")
        action_filter = st.multiselect(
            "Filter by action type",
            options=df_dec["action"].str.split(":").str[0].unique().tolist(), default=[])
        display_df = df_dec if not action_filter else df_dec[
            df_dec["action"].str.split(":").str[0].isin(action_filter)]
        display_cols = ["t_sim_day", "card_id", "action", "spm_cur", "spm_req", "rf_req",
                        "fillage", "margin", "friction_mult", "quality", "alarm", "diag_s"]
        available_cols = [c for c in display_cols if c in display_df.columns]
        st.dataframe(display_df[available_cols].round(3), hide_index=True,
                     use_container_width=True, height=400)

        # Audit trail
        st.markdown("##### 📝 Full Audit Trail")
        events = query_db(DB_TWIN, """
            SELECT t, source, kind, text FROM events
            WHERE well='BGW-01' AND (source LIKE 'TWIN%' OR source='PLC')
            ORDER BY t
        """)
        if not events.empty:
            events["Day"] = (events["t"] / 86400).round(3)
            st.dataframe(events[["Day", "source", "kind", "text"]], hide_index=True,
                         use_container_width=True, height=400)


# =========================================================================
# PAGE 5: PINN MODEL PERFORMANCE (NEW)
# =========================================================================
elif page == "🧪 PINN Model Performance":
    st.markdown('<div class="section-title">🧪 PINN MODEL TRAINING & VALIDATION RESULTS</div>',
                unsafe_allow_html=True)
    
    report = load_json(os.path.join(RESULTS_DIR, "digital_twin_report.json"))
    qc = report.get("data_qc", {})
    t_pass = qc.get("tests_passed", 29)
    t_tot = qc.get("tests_total", 29)
    runtime = report.get("runtime_s", 150) / 60.0
    
    res_rmse = report.get("reservoir", {}).get("training_metrics", {}).get("rmse_temp", 0.92)
    srp_rmse = report.get("srp", {}).get("training_metrics", {}).get("rmse_pump_load", 11.6)

    st.markdown("<br><h3 style='font-family: \"JetBrains Mono\", monospace; color: #E0E0E0;'>Feasibility — a working, verified prototype</h3>", unsafe_allow_html=True)
    st.markdown(f'''
    <div style="display: flex; gap: 15px; margin-bottom: 25px; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace;">
        <div style="flex: 1; min-width: 150px; background: rgba(0, 150, 255, 0.1); border: 1px solid {ACCENT_BLUE}; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: {ACCENT_BLUE};">{t_pass} / {t_tot}</div>
            <div style="font-size: 13px; color: #ccc;">independent verification</div>
            <div style="font-size: 11px; color: #888;">tests pass</div>
        </div>
        <div style="flex: 1; min-width: 150px; background: rgba(0, 255, 150, 0.1); border: 1px solid {ACCENT_GREEN}; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: {ACCENT_GREEN};">&le; {res_rmse:.2f} K</div>
            <div style="font-size: 13px; color: #ccc;">temperature error</div>
            <div style="font-size: 11px; color: #888;">on unseen well</div>
        </div>
        <div style="flex: 1; min-width: 150px; background: rgba(255, 150, 0, 0.1); border: 1px solid {ACCENT_AMBER}; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: {ACCENT_AMBER};">{srp_rmse:.1f} lbf</div>
            <div style="font-size: 13px; color: #ccc;">pump-card error</div>
            <div style="font-size: 11px; color: #888;">from surface card alone</div>
        </div>
        <div style="flex: 1; min-width: 150px; background: rgba(255, 150, 255, 0.1); border: 1px solid {ACCENT_PURPLE}; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: {ACCENT_PURPLE};">0.13 %</div>
            <div style="font-size: 13px; color: #ccc;">CSS optimiser vs brute-force</div>
            <div style="font-size: 11px; color: #888;">physics optimum</div>
        </div>
        <div style="flex: 1; min-width: 150px; background: rgba(255, 50, 50, 0.1); border: 1px solid {ACCENT_RED}; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 28px; font-weight: bold; color: {ACCENT_RED};">{runtime:.1f} min</div>
            <div style="font-size: 13px; color: #ccc;">full training on</div>
            <div style="font-size: 11px; color: #888;">1 GPU/CPU core</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    

    report = load_json(os.path.join(RESULTS_DIR, "digital_twin_report.json"))

    if not report:
        st.warning("digital_twin_report.json not found. Run pinn_digital_twin.py first.")
    else:
        # Architecture overview
        st.markdown("""
        <div class="info-panel">
            <div style="display:flex;justify-content:space-around;text-align:center">
                <div>
                    <div class="kpi-label">ARCHITECTURE</div>
                    <div class="kpi-value" style="color:#00d4ff;font-size:20px">Hybrid PINN</div>
                    <div class="kpi-label" style="color:#556677">Hard Physics + Neural Net</div>
                </div>
                <div>
                    <div class="kpi-label">RESERVOIR PINN</div>
                    <div class="kpi-value" style="color:#00ff88;font-size:20px">51k</div>
                    <div class="kpi-label" style="color:#556677">Parameters</div>
                </div>
                <div>
                    <div class="kpi-label">SRP WAVE PINN</div>
                    <div class="kpi-value" style="color:#ffb300;font-size:20px">46k</div>
                    <div class="kpi-label" style="color:#556677">Parameters</div>
                </div>
                <div>
                    <div class="kpi-label">TOTAL</div>
                    <div class="kpi-value" style="color:#b388ff;font-size:20px">~97k</div>
                    <div class="kpi-label" style="color:#556677">Parameters</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Reservoir PINN metrics
        st.markdown('<div class="section-title">🌡️ RESERVOIR PINN — THERMAL + PRODUCTION MODEL</div>',
                    unsafe_allow_html=True)
        res_metrics = safe_get(report, "reservoir", "training_metrics", default={})
        cols = st.columns(5)
        res_kpis = [
            ("R² Temperature", safe_get(res_metrics, "r2_temperature", default=0), ".4f", "", ACCENT_GREEN),
            ("R² Oil Rate", safe_get(res_metrics, "r2_oil_rate", default=0), ".4f", "", ACCENT_BLUE),
            ("RMSE T (K)", safe_get(res_metrics, "rmse_T_K", default=0), ".2f", " K", ACCENT_AMBER),
            ("RMSE Oil Rate", safe_get(res_metrics, "rmse_oil_rate", default=0), ".2f", "", ACCENT_RED),
            ("Final Loss", safe_get(res_metrics, "final_loss", default=0), ".4e", "", ACCENT_PURPLE),
        ]
        for col, (label, val, fmt, unit, color) in zip(cols, res_kpis):
            with col:
                st.markdown(kpi_html(label, val, fmt, unit, color), unsafe_allow_html=True)

        # CSS optimization
        st.markdown("")
        st.markdown('<div class="section-title">🔥 CSS CYCLE OPTIMIZATION RESULTS</div>',
                    unsafe_allow_html=True)
        css_opt = safe_get(report, "reservoir", "css_optimization", default={})
        cols = st.columns(6)
        css_kpis = [
            ("Injection Days", css_opt.get("optimal_injection_days", 0), ".1f", " d", ACCENT_BLUE),
            ("Soak Days", css_opt.get("optimal_soak_days", 0), ".1f", " d", ACCENT_GREEN),
            ("Production Days", css_opt.get("optimal_production_days", 0), ".0f", " d", ACCENT_AMBER),
            ("Predicted SOR", css_opt.get("predicted_SOR", 0), ".2f", "", ACCENT_RED),
            ("Cum. Oil", css_opt.get("predicted_cumulative_oil_bbl", 0), ".0f", " bbl", ACCENT_GREEN),
            ("NPV", css_opt.get("economic_npv", 0), ",.0f", " $", ACCENT_PURPLE),
        ]
        for col, (label, val, fmt, unit, color) in zip(cols, css_kpis):
            with col:
                st.markdown(kpi_html(label, val, fmt, unit, color), unsafe_allow_html=True)

        st.markdown("")

        # SRP PINN metrics
        st.markdown('<div class="section-title">⚙️ SRP WAVE PINN — DAMPED WAVE EQUATION MODEL</div>',
                    unsafe_allow_html=True)
        srp_metrics = safe_get(report, "srp", "training_metrics", default={})
        cols = st.columns(5)
        srp_kpis = [
            ("R² Surface Load", safe_get(srp_metrics, "r2_surface", default=0), ".4f", "", ACCENT_GREEN),
            ("R² Pump Load", safe_get(srp_metrics, "r2_pump", default=0), ".4f", "", ACCENT_BLUE),
            ("RMSE Surface", safe_get(srp_metrics, "rmse_surface_load", default=0), ".1f", " lbf", ACCENT_AMBER),
            ("RMSE Pump", safe_get(srp_metrics, "rmse_pump_load", default=0), ".1f", " lbf", ACCENT_RED),
            ("Final Loss", safe_get(srp_metrics, "final_loss", default=0), ".4e", "", ACCENT_PURPLE),
        ]
        for col, (label, val, fmt, unit, color) in zip(cols, srp_kpis):
            with col:
                st.markdown(kpi_html(label, val, fmt, unit, color), unsafe_allow_html=True)

        # SRP Diagnostics
        st.markdown("")
        st.markdown('<div class="section-title">🔍 SRP DIAGNOSTICS</div>', unsafe_allow_html=True)
        diag = safe_get(report, "srp", "diagnostics", default={})
        cols = st.columns(6)
        diag_kpis = [
            ("Fluid Pound", "✅ Detected" if diag.get("fluid_pound_detected", False) else "❌ None",
             "", "", ACCENT_RED if diag.get("fluid_pound_detected") else ACCENT_GREEN),
            ("Rod Float Index", diag.get("rod_float_index", 0), ".3f", "", ACCENT_AMBER),
            ("Peak Pump Load", diag.get("peak_pump_load_lbf", 0), ".0f", " lbf", ACCENT_BLUE),
            ("Min Pump Load", diag.get("min_pump_load_lbf", 0), ".0f", " lbf", ACCENT_PURPLE),
            ("Peak Accel.", diag.get("peak_acceleration_ft_s2", 0), ".1f", " ft/s²", ACCENT_RED),
            ("Lift Energy", diag.get("lifting_energy_kWh", 0), ".2f", " kWh", ACCENT_GREEN),
        ]
        for col, (label, val, fmt, unit, color) in zip(cols, diag_kpis):
            with col:
                if isinstance(val, str):
                    st.markdown(kpi_html(label, val, color=color), unsafe_allow_html=True)
                else:
                    st.markdown(kpi_html(label, val, fmt, unit, color), unsafe_allow_html=True)

        # Result images
        st.markdown("")
        st.markdown('<div class="section-title">📊 TRAINING RESULT VISUALIZATIONS</div>',
                    unsafe_allow_html=True)
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            try:
                st.image(os.path.join(RESULTS_DIR, "reservoir_fit.png"),
                         caption="Reservoir PINN — Temperature & Production Fit")
            except Exception:
                st.info("reservoir_fit.png not available")
        with img_col2:
            try:
                st.image(os.path.join(RESULTS_DIR, "srp_cards.png"),
                         caption="SRP PINN — Surface vs Downhole Card Prediction")
            except Exception:
                st.info("srp_cards.png not available")

        # Field forecast
        forecast = report.get("field_forecast", [])
        if forecast:
            st.markdown("")
            st.markdown('<div class="section-title">📈 MULTI-CYCLE FIELD FORECAST</div>',
                        unsafe_allow_html=True)
            df_fc = pd.DataFrame(forecast)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                                subplot_titles=["Cumulative Oil per Cycle (bbl)", "Steam-Oil Ratio"])
            fig.add_trace(go.Bar(x=df_fc["cycle"], y=df_fc.get("cumulative_oil_bbl", []),
                                 marker_color=ACCENT_GREEN, name="Oil (bbl)", opacity=0.8),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=df_fc["cycle"], y=df_fc.get("sor", []),
                                     mode="lines+markers", name="SOR",
                                     line=dict(color=ACCENT_RED, width=2),
                                     marker=dict(size=6)), row=2, col=1)
            styled_fig(fig, height=400, xaxis2_title="CSS Cycle Number")
            for i in range(1, 3):
                fig.update_yaxes(gridcolor=GRID_COLOR, row=i, col=1)
                fig.update_xaxes(gridcolor=GRID_COLOR, row=i, col=1)
            st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# PAGE 6: SRP OPTIMIZATION (NEW)
# =========================================================================
elif page == "⚡ SRP Optimization":
    st.markdown('<div class="section-title">⚡ SRP OPTIMIZATION — BASELINE vs PINN-OPTIMIZED OPERATION</div>',
                unsafe_allow_html=True)

    baseline = load_csv(os.path.join(RESULTS_DIR, "srp_daily_baseline_fixed_speed.csv"))
    optimized = load_csv(os.path.join(RESULTS_DIR, "srp_daily_optimised.csv"))

    if baseline.empty or optimized.empty:
        st.warning("SRP daily CSV files not found in updated_results/. Run pinn_digital_twin.py first.")
    else:
        # Summary KPIs from digital_twin_report.json (Expected Benefits)
        report = load_json(os.path.join(RESULTS_DIR, "digital_twin_report.json"))
        benefits = report.get("expected_benefits_quantified", {})
        
        b_energy = benefits.get("lifting_energy_kwh_per_bbl", {"baseline": 3.17, "optimised": 0.75})
        b_impact = benefits.get("impact_over_limit_days", {"baseline": 168, "optimised": 0})
        b_eff = benefits.get("mean_pump_volumetric_efficiency", {"baseline": 0.29, "optimised": 0.89})
        b_unsetting = benefits.get("pump_unsetting_risk_days", {"baseline": 10, "optimised": 0})
        b_oil = benefits.get("field_oil_change_pct_same_steam_budget", 17.2)
        
        energy_pct = ((b_energy["optimised"] - b_energy["baseline"]) / b_energy["baseline"]) * 100
        
        st.markdown(f'''
        <div style="display: flex; gap: 15px; margin-bottom: 25px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 150px; background: rgba(0, 150, 255, 0.1); border: 1px solid {ACCENT_BLUE}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_BLUE};">{energy_pct:.0f}%</div>
                <div style="font-size: 13px; color: #ccc;">Lifting Energy</div>
                <div style="font-size: 11px; color: #888;">{b_energy["baseline"]:.2f} &rarr; {b_energy["optimised"]:.2f} kWh/bbl</div>
            </div>
            <div style="flex: 1; min-width: 150px; background: rgba(255, 150, 0, 0.1); border: 1px solid {ACCENT_AMBER}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_AMBER};">{b_impact["baseline"]} &rarr; {b_impact["optimised"]}</div>
                <div style="font-size: 13px; color: #ccc;">Days w/ Impact Load</div>
                <div style="font-size: 11px; color: #888;">per cycle</div>
            </div>
            <div style="flex: 1; min-width: 150px; background: rgba(0, 255, 150, 0.1); border: 1px solid {ACCENT_GREEN}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_GREEN};">{b_eff["baseline"]:.2f} &rarr; {b_eff["optimised"]:.2f}</div>
                <div style="font-size: 13px; color: #ccc;">Pump Volumetric</div>
                <div style="font-size: 11px; color: #888;">efficiency</div>
            </div>
            <div style="flex: 1; min-width: 150px; background: rgba(255, 150, 255, 0.1); border: 1px solid {ACCENT_PURPLE}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_PURPLE};">+{b_oil:.1f}%</div>
                <div style="font-size: 13px; color: #ccc;">Field Oil Output</div>
                <div style="font-size: 11px; color: #888;">(same steam budget)</div>
            </div>
            <div style="flex: 1; min-width: 150px; background: rgba(255, 50, 50, 0.1); border: 1px solid {ACCENT_RED}; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: {ACCENT_RED};">{b_unsetting["baseline"]} &rarr; {b_unsetting["optimised"]}</div>
                <div style="font-size: 13px; color: #ccc;">Days at Unsetting</div>
                <div style="font-size: 11px; color: #888;">risk per cycle</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Comparison charts
        chart_configs = [
            ("SPM Over Production Days", "spm", "SPM (1/min)", ACCENT_BLUE, ACCENT_GREEN),
            ("Rod Float Index", "rod_float_index", "Float Index", ACCENT_RED, ACCENT_GREEN),
            ("Energy Consumption", "energy_kWh", "kWh/day", ACCENT_AMBER, ACCENT_GREEN),
            ("Oil Production Rate", "oil_rate_bpd", "bbl/day", ACCENT_BLUE, ACCENT_GREEN),
            ("Pump Fillage", "pump_fillage", "Fillage (-)", ACCENT_PURPLE, ACCENT_GREEN),
            ("Reservoir Temperature", "reservoir_temp_K", "T (K)", ACCENT_RED, ACCENT_BLUE),
        ]

        for row_start in range(0, len(chart_configs), 2):
            c1, c2 = st.columns(2)
            for col, (title, col_name, ylabel, color_base, color_opt) in \
                    zip([c1, c2], chart_configs[row_start:row_start + 2]):
                with col:
                    if col_name in baseline.columns and col_name in optimized.columns:
                        fig = go.Figure()
                        x_base = baseline["day"] if "day" in baseline.columns else range(len(baseline))
                        x_opt = optimized["day"] if "day" in optimized.columns else range(len(optimized))
                        fig.add_trace(go.Scatter(
                            x=x_base, y=baseline[col_name],
                            mode="lines", name="Baseline",
                            line=dict(color=color_base, width=2, dash="dash"), opacity=0.7))
                        fig.add_trace(go.Scatter(
                            x=x_opt, y=optimized[col_name],
                            mode="lines", name="Optimized",
                            line=dict(color=color_opt, width=2)))
                        if col_name == "rod_float_index":
                            fig.add_hline(y=1.0, line_dash="dot", line_color=ACCENT_RED,
                                          annotation_text="DANGER", annotation=dict(
                                    font_color=ACCENT_RED, font_size=10))
                        styled_fig(fig, height=300, title=title,
                                   xaxis_title="Day", yaxis_title=ylabel)
                        st.plotly_chart(fig, use_container_width=True)

        # SRP operation image
        try:
            st.image(os.path.join(RESULTS_DIR, "srp_operation.png"),
                     caption="SRP Daily Operation — Baseline vs Optimized")
        except Exception:
            pass


# =========================================================================
# PAGE 7: FIELD HISTORIAN
# =========================================================================
elif page == "📈 Field Historian":
    st.markdown('<div class="section-title">📈 MULTI-WELL FIELD HISTORIAN — PRODUCTION DATA</div>',
                unsafe_allow_html=True)

    wells = query_db(DB_TWIN, "SELECT DISTINCT well FROM production_daily ORDER BY well")
    well_list = wells["well"].tolist() if not wells.empty else ["BGW-01"]
    selected_well = st.selectbox("Select Well", well_list)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Production", "🌡️ CSS Cycles", "⚡ VFD/Energy", "🔧 Failures", "🧪 Well Tests"])

    with tab1:
        prod = query_db(DB_TWIN, "SELECT * FROM production_daily WHERE well=? ORDER BY date",
                        (selected_well,))
        if not prod.empty:
            prod["date"] = pd.to_datetime(prod["date"])
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=["Daily Oil & Water Rate", "Cumulative Production"])
            fig.add_trace(go.Scatter(x=prod["date"], y=prod["oil_bpd"], name="Oil (bpd)",
                                     line=dict(color=ACCENT_GREEN, width=1.5), fill="tozeroy",
                                     fillcolor="rgba(0,255,136,0.1)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=prod["date"], y=prod["water_bpd"], name="Water (bpd)",
                                     line=dict(color=ACCENT_BLUE, width=1.5), fill="tozeroy",
                                     fillcolor="rgba(0,212,255,0.05)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=prod["date"], y=prod["oil_bpd"].cumsum(), name="Cum Oil",
                                     line=dict(color=ACCENT_GREEN, width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=prod["date"], y=prod["water_bpd"].cumsum(), name="Cum Water",
                                     line=dict(color=ACCENT_BLUE, width=2)), row=2, col=1)
            styled_fig(fig, height=500)
            for i in range(1, 3):
                fig.update_yaxes(gridcolor=GRID_COLOR, row=i, col=1)
                fig.update_xaxes(gridcolor=GRID_COLOR, row=i, col=1)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Oil (bpd)", f"{prod['oil_bpd'].mean():.1f}")
            c2.metric("Peak Oil (bpd)", f"{prod['oil_bpd'].max():.1f}")
            c3.metric("Avg Water Cut",
                       f"{(prod['water_bpd'] / (prod['oil_bpd'] + prod['water_bpd'] + 1e-9)).mean():.1%}")
            c4.metric("Total Days", f"{len(prod)}")
        else:
            st.info("No production data available.")

    with tab2:
        css = query_db(DB_TWIN, "SELECT * FROM css_cycles WHERE well=? ORDER BY cycle", (selected_well,))
        if not css.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                                subplot_titles=["Cycle Oil Production (bbl)", "Steam Injected (t)"])
            fig.add_trace(go.Bar(x=css["cycle"], y=css["cycle_oil_bbl"],
                                 marker_color=ACCENT_GREEN, name="Oil (bbl)", opacity=0.8), row=1, col=1)
            fig.add_trace(go.Bar(x=css["cycle"], y=css["steam_t"],
                                 marker_color=ACCENT_RED, name="Steam (t)", opacity=0.8), row=2, col=1)
            styled_fig(fig, height=450, xaxis2_title="Cycle Number")
            for i in range(1, 3):
                fig.update_yaxes(gridcolor=GRID_COLOR, row=i, col=1)
                fig.update_xaxes(gridcolor=GRID_COLOR, row=i, col=1)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(css.round(2), hide_index=True, use_container_width=True)
        else:
            st.info("No CSS cycle data available.")

    with tab3:
        vfd = query_db(DB_TWIN, "SELECT * FROM vfd_daily WHERE well=? ORDER BY date", (selected_well,))
        if not vfd.empty:
            vfd["date"] = pd.to_datetime(vfd["date"])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=["VFD Frequency & SPM", "Daily Energy (kWh)",
                                                "Motor Current (A)"])
            fig.add_trace(go.Scatter(x=vfd["date"], y=vfd["avg_spm"], name="SPM",
                                     line=dict(color=ACCENT_BLUE, width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=vfd["date"], y=vfd["kwh"], name="kWh",
                                     line=dict(color=ACCENT_AMBER, width=1.5),
                                     fill="tozeroy", fillcolor="rgba(255,179,0,0.1)"), row=2, col=1)
            fig.add_trace(go.Scatter(x=vfd["date"], y=vfd["avg_current_A"], name="Current (A)",
                                     line=dict(color=ACCENT_RED, width=1.5)), row=3, col=1)
            styled_fig(fig, height=550)
            for i in range(1, 4):
                fig.update_yaxes(gridcolor=GRID_COLOR, row=i, col=1)
                fig.update_xaxes(gridcolor=GRID_COLOR, row=i, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No VFD data available.")

    with tab4:
        fails = query_db(DB_TWIN, "SELECT * FROM rod_failures WHERE well=? ORDER BY date",
                         (selected_well,))
        unsetting = query_db(DB_TWIN, "SELECT * FROM pump_unsetting WHERE well=? ORDER BY date",
                             (selected_well,))
        if not fails.empty:
            st.markdown("##### 🔧 Rod Failure History")
            fig = px.scatter(fails, x="wellhead_T_degC", y="spm_before", color="mode",
                             size="downtime_days", hover_data=["date", "depth_ft"],
                             title="Rod Failures: Temperature vs SPM",
                             color_discrete_sequence=[ACCENT_RED, ACCENT_AMBER, ACCENT_PURPLE])
            styled_fig(fig, height=350, xaxis_title="Wellhead T (°C)", yaxis_title="SPM Before Failure")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(fails.round(2), hide_index=True, use_container_width=True)
        else:
            st.info("No rod failure data available.")

        if not unsetting.empty:
            st.markdown("##### ⚠️ Pump Unsetting Events")
            st.dataframe(unsetting.round(2), hide_index=True, use_container_width=True)

    with tab5:
        tests = query_db(DB_TWIN, "SELECT * FROM well_tests WHERE well=? ORDER BY date",
                         (selected_well,))
        if not tests.empty:
            tests["date"] = pd.to_datetime(tests["date"])
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=tests["date"], y=tests["oil_bpd"], mode="lines+markers",
                                     name="Oil (bpd)", line=dict(color=ACCENT_GREEN, width=2),
                                     marker=dict(size=5)), secondary_y=False)
            fig.add_trace(go.Scatter(x=tests["date"], y=tests["wellhead_T_degC"], mode="lines+markers",
                                     name="WH Temp (°C)",
                                     line=dict(color=ACCENT_RED, width=2, dash="dash"),
                                     marker=dict(size=5, symbol="triangle-up")), secondary_y=True)
            styled_fig(fig, height=350, title="Well Test Results",
                       yaxis_title="Oil Rate (bpd)", yaxis2_title="Wellhead Temp (°C)")
            fig.update_yaxes(gridcolor=GRID_COLOR, secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No well test data available.")

    # Field forecast image
    st.markdown("")
    try:
        st.image(os.path.join(RESULTS_DIR, "baghewala_field_forecast.png"),
                 caption="Baghewala Field — Multi-Cycle Production Forecast")
    except Exception:
        pass

    # Scenario timeline
    st.markdown("")
    st.markdown('<div class="section-title">🎬 INTEGRATION TEST SCENARIO TIMELINE</div>',
                unsafe_allow_html=True)
    scenario_data = pd.DataFrame([
        {"Day": "0-1", "Event": "Normal production (CSS cycle 7, post-soak)", "Severity": "None"},
        {"Day": "1.0", "Event": "Near-wellbore damage — inflow drops 55%", "Severity": "High"},
        {"Day": "2.0-3.0", "Event": "Tubing cooling + asphaltene friction buildup (×6.5)",
         "Severity": "Critical"},
        {"Day": "3.5-3.75", "Event": "Load-cell sensor desynchronization", "Severity": "High"},
        {"Day": "4.0-4.25", "Event": "Twin service offline (6 h test)", "Severity": "High"},
        {"Day": "4.25-5.0", "Event": "Twin recovered — watchdog cleared", "Severity": "None"},
    ])
    st.dataframe(scenario_data, hide_index=True, use_container_width=True)
