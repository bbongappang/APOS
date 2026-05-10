"""
A-APOS v3.0 — app.py
Multi-tab Streamlit dashboard with LLM assistant panel
"""
import streamlit as st
import streamlit.components.v1 as components
import json, time, os, simpy
import pandas as pd
import numpy as np

st.set_page_config(
    layout="wide",
    page_title="A-APOS Factory OS v3.0",
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(BASE_DIR, "A_APOS_Engine")
DATA_PATH  = os.path.join(BASE_DIR, "SMT_2020 - Final", "AutoSched")

BASELINE_MAP = {1: 949, 2: 897, 3: 923, 4: 955}
DS_LABELS = {
    1: "DS1 · HVLM (소품종 대량, 고장 없음)",
    2: "DS2 · LVHM (다품종 소량, 고장 없음)",
    3: "DS3 · HVLM_E (소품종, 고장 포함)",
    4: "DS4 · LVHM_E (다품종, 고장 포함)",
}
DISASTER_SCENARIOS = {
    "A — Litho 48h 전면 다운": {"area": "Litho",     "duration": 2880, "desc": "Litho 구역 전 설비 48시간 셧다운"},
    "B — Diffusion 가스 공급 중단": {"area": "Diffusion", "duration": 4320, "desc": "Diffusion 구역 72시간 가동 중단"},
    "C — 복합 재난 (Litho + HotLot 50개 긴급)": {"area": "Litho", "duration": 2880, "desc": "Litho 다운 + HotLot 50개 긴급 투입"},
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Font & base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background: #f8fafc; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: #94a3b8 !important; font-size: 12px; }

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"] { font-size: 11px !important; color: #64748b !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700 !important; color: #0f172a !important; }

/* Tab styling */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: white;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #e2e8f0;
    gap: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    font-size: 13px;
    color: #64748b;
    padding: 8px 18px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #2563eb !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    border: 1px solid #e2e8f0;
    transition: all 0.15s;
}
.stButton > button[kind="primary"] {
    background: #2563eb;
    border-color: #2563eb;
    color: white;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }

/* KPI banner */
.kpi-banner {
    display: flex; gap: 12px; margin-bottom: 16px;
}
.kpi-card {
    flex: 1; background: white; border-radius: 12px;
    padding: 16px 20px; border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border-top: 3px solid var(--accent, #2563eb);
}
.kpi-label { font-size: 10px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #0f172a; line-height: 1; }
.kpi-sub   { font-size: 11px; color: #64748b; margin-top: 4px; }

/* Alert */
.alert-red  { background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 10px 14px; color: #991b1b; font-size: 13px; }
.alert-amber{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 10px 14px; color: #92400e; font-size: 13px; }
.alert-green{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 14px; color: #14532d; font-size: 13px; }
.alert-blue { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 14px; color: #1e40af; font-size: 13px; }

/* Section header */
.sec-header {
    font-size: 13px; font-weight: 600; color: #374151;
    margin: 16px 0 10px; padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
    display: flex; align-items: center; gap: 8px;
}

/* LLM chat bubble */
.chat-user { background: #2563eb; color: white; border-radius: 12px 12px 2px 12px; padding: 10px 14px; margin: 6px 0 6px 40px; font-size: 13px; }
.chat-ai   { background: white; color: #1e293b; border: 1px solid #e2e8f0; border-radius: 12px 12px 12px 2px; padding: 10px 14px; margin: 6px 40px 6px 0; font-size: 13px; line-height: 1.6; }

/* Policy badge */
.badge-best  { background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.badge-worst { background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.badge-mid   { background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

# ── Imports (lazy, with error handling) ──────────────────────────────────────
try:
    from A_APOS_Engine.data_manager import APOSDataManager
    from A_APOS_Engine.engine_wrapper import SimBridge
    ENGINE_OK = True
except Exception as e:
    ENGINE_OK = False
    ENGINE_ERR = str(e)

# ── Session init ──────────────────────────────────────────────────────────────
def init_session(ds_id: int, overrides: dict = None):
    if not ENGINE_OK:
        return
    try:
        dm   = APOSDataManager(base_path=DATA_PATH)
        data = dm.load_dataset(ds_id)
        env  = simpy.Environment()
        bridge = SimBridge(env, data, overrides=overrides or {})
        st.session_state.update({
            "dm": dm, "ds_id": ds_id, "data": data,
            "env": env, "bridge": bridge, "tick": 0,
            "running": False, "kpi_log": [],
            "gnn_log_history": [], "overrides": overrides or {},
        })
    except Exception as e:
        st.session_state["init_error"] = str(e)

for key, default in [
    ("ds_id", 4), ("tick", 0), ("running", False),
    ("kpi_log", []), ("gnn_log_history", []),
    ("benchmark_data", {}), ("disaster_log", []),
    ("chat_history", []), ("overrides", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "bridge" not in st.session_state and ENGINE_OK:
    init_session(4)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 12px; border-bottom:1px solid #1e293b; margin-bottom:16px;">
        <div style="font-size:20px; font-weight:700; color:#f1f5f9; letter-spacing:0.02em;">🏭 A-APOS</div>
        <div style="font-size:11px; color:#475569; margin-top:2px;">Factory OS v3.0 · SMT 2020</div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset
    st.markdown('<div style="font-size:11px;color:#475569;font-weight:600;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.06em;">📂 Dataset</div>', unsafe_allow_html=True)
    ds_choice = st.selectbox("", [1,2,3,4], index=st.session_state.ds_id-1,
                              format_func=lambda x: DS_LABELS[x], label_visibility="collapsed")
    if ds_choice != st.session_state.ds_id:
        init_session(ds_choice)
        st.rerun()

    st.markdown("---")

    # Simulation control
    st.markdown('<div style="font-size:11px;color:#475569;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em;">⚙️ Simulation Control</div>', unsafe_allow_html=True)
    sim_speed = st.slider("Step Size (분)", 10, 500, 50, step=10)
    policy = st.selectbox("Dispatching Policy", ["GNN","FIFO","EDD","CR"],
                           index=["GNN","FIFO","EDD","CR"].index(st.session_state.overrides.get("policy","GNN")))
    wip_limit = st.number_input("WIP Limit", 1000, 10000,
                                 st.session_state.overrides.get("wip_limit", 3000), step=500)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ 시작", use_container_width=True, type="primary"):
            st.session_state.running = True
    with c2:
        if st.button("⏹ 중단", use_container_width=True):
            st.session_state.running = False

    if st.button("🔄 초기화", use_container_width=True):
        init_session(st.session_state.ds_id, {"policy": policy, "wip_limit": wip_limit})
        st.rerun()

    if st.button("🚀 정책 적용", use_container_width=True):
        init_session(st.session_state.ds_id, {"policy": policy, "wip_limit": wip_limit})
        st.success(f"{policy} 정책 적용됨")
        st.rerun()

    st.markdown("---")

    # Live KPI
    st.markdown('<div style="font-size:11px;color:#475569;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em;">📊 Live KPI</div>', unsafe_allow_html=True)
    st.metric("Sim Time", f"T+{st.session_state.tick}")

    if ENGINE_OK and "bridge" in st.session_state:
        try:
            s = st.session_state.bridge.get_summary()
            st.metric("WIP (재공품)", f"{s['wip']:,}")
            st.metric("완료 Lot", f"{s['completed']:,}")
            kh = st.session_state.bridge.kpi_history
            if kh:
                st.metric("Avg CT", f"{kh[-1]['ct']:.0f} h")
                st.metric("OTD", f"{kh[-1]['ontime']:.1f} %")
        except:
            pass

    st.markdown("---")
    st.caption(f"Baseline LT: {BASELINE_MAP[st.session_state.ds_id]}h")

# ── Main layout: tabs + LLM panel ────────────────────────────────────────────
main_col, llm_col = st.columns([3.2, 0.8])

with main_col:
    # Top header bar
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                background:white;border-radius:12px;padding:14px 20px;
                border:1px solid #e2e8f0;margin-bottom:16px;
                box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <div>
            <div style="font-size:20px;font-weight:700;color:#0f172a;">A-APOS Factory Operating System</div>
            <div style="font-size:12px;color:#64748b;margin-top:2px;">
                {DS_LABELS[st.session_state.ds_id]} &nbsp;|&nbsp; 
                Policy: <b>{st.session_state.overrides.get('policy','GNN')}</b> &nbsp;|&nbsp;
                Baseline LT: {BASELINE_MAP[st.session_state.ds_id]}h
            </div>
        </div>
        <div style="font-size:12px;color:#94a3b8;text-align:right;">
            SMT 2020<br>GNN + XGBoost
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get current state
    cur = {}
    if ENGINE_OK and "bridge" in st.session_state:
        try:
            cur = st.session_state.bridge.update_ui_state()
            kh  = st.session_state.bridge.kpi_history
            # accumulate benchmark
            if kh:
                pol = st.session_state.overrides.get("policy","GNN")
                st.session_state.benchmark_data[pol] = {"ct": kh[-1]["ct"], "ontime": kh[-1]["ontime"],
                                                         "wip": cur.get("wip",0), "cqt": cur.get("cqt",{}).get("violations",0)}
        except:
            pass

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab_main, tab_sim, tab_bench, tab_disaster, tab_xai = st.tabs([
        "🏠  Overview",
        "📡  Simulation",
        "🏆  Benchmark Arena",
        "🚨  재난 시뮬레이터",
        "🔍  AI Explainability",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ════════════════════════════════════════════════════════
    with tab_main:
        # KPI row
        kpi = cur.get("kpi", {})
        kh  = st.session_state.get("bridge") and st.session_state.bridge.kpi_history or []
        last_kh = kh[-1] if kh else {}

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("WIP (재공품)", f"{cur.get('wip',0):,}", help="현재 공장 내 총 재공품 수")
        with c2: st.metric("완료 Lot", f"{kpi.get('completed',0):,}")
        with c3: st.metric("Avg Cycle Time", f"{last_kh.get('ct',0):.0f} h",
                            delta=f"{last_kh.get('ct',0)-BASELINE_MAP[st.session_state.ds_id]:.0f}h vs baseline",
                            delta_color="inverse")
        with c4: st.metric("OTD (납기준수율)", f"{last_kh.get('ontime',0):.1f} %")
        with c5: st.metric("CQT 위반", f"{cur.get('cqt',{}).get('violations',0)}")

        st.markdown('<div class="sec-header">📍 Area Health Map</div>', unsafe_allow_html=True)

        areas = [
            {"name":"Diffusion","mttf":10080,"mttr":151.2},
            {"name":"Dry_Etch","mttf":10080,"mttr":231.84},
            {"name":"Litho","mttf":10080,"mttr":705.59},
            {"name":"Implant","mttf":10080,"mttr":604.8},
            {"name":"Dielectric","mttf":10080,"mttr":604.8},
            {"name":"Planar","mttf":10080,"mttr":201.6},
            {"name":"TF","mttf":10080,"mttr":453.6},
            {"name":"Wet_Etch","mttf":10080,"mttr":221.76},
            {"name":"Def_Met","mttf":10080,"mttr":35.28},
            {"name":"Litho_Met","mttf":10080,"mttr":35.28},
            {"name":"TF_Met","mttf":10080,"mttr":35.28},
        ]
        area_stats = cur.get("area_stats", {})
        xgb_probs  = cur.get("xgb_probs", {})

        cols = st.columns(4)
        for i, a in enumerate(areas):
            stats = area_stats.get(a["name"], {})
            total = max(stats.get("total",1), 1)
            util  = stats.get("avg_util", round(stats.get("busy",0)/total*100, 1))
            down  = stats.get("down", 0)
            wip   = stats.get("wip", 0)
            prob  = xgb_probs.get(a["name"], 0)
            avail = round(a["mttf"]/(a["mttf"]+a["mttr"])*100, 1)

            if prob >= 0.7 or down >= 2:
                border_color = "#ef4444"
                bg = "#fef2f2"
                status_icon = "🔴"
            elif prob >= 0.4 or down >= 1:
                border_color = "#f59e0b"
                bg = "#fffbeb"
                status_icon = "🟡"
            else:
                border_color = "#22c55e"
                bg = "#f0fdf4"
                status_icon = "🟢"

            util_bar = f'<div style="background:#e2e8f0;border-radius:3px;height:5px;margin:6px 0;"><div style="width:{min(util,100)}%;background:{border_color};height:5px;border-radius:3px;"></div></div>'

            html = f"""
            <div style="background:{bg};border:1px solid {border_color};border-radius:10px;
                        padding:12px;margin-bottom:8px;border-left:4px solid {border_color};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="font-size:12px;font-weight:600;color:#1e293b;">{status_icon} {a['name']}</div>
                    <div style="font-size:11px;font-weight:700;color:{border_color};">{prob*100:.0f}%</div>
                </div>
                {util_bar}
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#64748b;">
                    <span>Util {util:.0f}%</span>
                    <span>WIP {wip}</span>
                    <span style="color:{'#ef4444' if down>0 else '#94a3b8'}">Down {down}</span>
                </div>
                <div style="font-size:9px;color:#94a3b8;margin-top:4px;">Avail {avail}% · MTTR {a['mttr']}min</div>
            </div>
            """
            with cols[i % 4]:
                st.markdown(html, unsafe_allow_html=True)

        # XGBoost bottleneck warning
        high_prob = [(a, round(xgb_probs[a]*100)) for a in xgb_probs if xgb_probs[a] >= 0.7]
        if high_prob:
            warnings = ", ".join([f"<b>{a}</b> ({p}%)" for a, p in high_prob])
            st.markdown(f'<div class="alert-red">🚨 <b>Bottleneck Alert</b> — XGBoost 예측: {warnings} 구역 위험 (≥70%)</div>', unsafe_allow_html=True)

        # CQT urgent lots
        cqt = cur.get("cqt", {})
        urgent_list = cqt.get("urgent_list", [])
        if urgent_list:
            st.markdown('<div class="sec-header">⚠️ CQT Urgency — 긴급 Lot 현황</div>', unsafe_allow_html=True)
            df_cqt = pd.DataFrame(urgent_list)
            if not df_cqt.empty:
                st.dataframe(df_cqt, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # TAB 2: SIMULATION
    # ════════════════════════════════════════════════════════
    with tab_sim:
        st.markdown('<div class="sec-header">📡 Real-time Simulation State</div>', unsafe_allow_html=True)

        # WIP trend
        wip_hist = cur.get("wip_history", [])
        if wip_hist:
            df_wip = pd.DataFrame(wip_hist)
            import plotly.graph_objects as go
            fig_wip = go.Figure()
            fig_wip.add_trace(go.Scatter(x=df_wip["tick"], y=df_wip["wip"],
                                          mode="lines", name="WIP",
                                          line=dict(color="#2563eb", width=2),
                                          fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
            if "limit" in df_wip.columns:
                fig_wip.add_trace(go.Scatter(x=df_wip["tick"], y=df_wip["limit"],
                                              mode="lines", name="WIP Limit (Floodgate)",
                                              line=dict(color="#ef4444", width=1.5, dash="dash")))
            fig_wip.update_layout(
                title="WIP 추이 & Floodgate 상한",
                xaxis_title="Sim Time (min)", yaxis_title="WIP (lots)",
                height=260, margin=dict(l=40,r=20,t=40,b=40),
                legend=dict(orientation="h", y=1.1),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#f1f5f9"), yaxis=dict(gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig_wip, use_container_width=True)
        else:
            st.info("시뮬레이션을 시작하면 WIP 추이가 표시됩니다.")

        # KPI trend
        kpi_hist = cur.get("kpi_history", [])
        if kpi_hist:
            df_kpi = pd.DataFrame(kpi_hist)
            col_ct, col_ot = st.columns(2)
            with col_ct:
                fig_ct = go.Figure()
                fig_ct.add_trace(go.Scatter(x=df_kpi["tick"], y=df_kpi["ct"],
                                             mode="lines+markers", name="Avg CT",
                                             line=dict(color="#7c3aed", width=2)))
                fig_ct.add_hline(y=BASELINE_MAP[st.session_state.ds_id],
                                  line_dash="dot", line_color="#94a3b8",
                                  annotation_text="Baseline")
                fig_ct.update_layout(title="Avg Cycle Time (h)", height=220,
                                      margin=dict(l=40,r=20,t=40,b=40),
                                      plot_bgcolor="white", paper_bgcolor="white",
                                      yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="#f1f5f9"))
                st.plotly_chart(fig_ct, use_container_width=True)
            with col_ot:
                fig_ot = go.Figure()
                fig_ot.add_trace(go.Scatter(x=df_kpi["tick"], y=df_kpi["ontime"],
                                             mode="lines+markers", name="OTD %",
                                             line=dict(color="#059669", width=2),
                                             fill="tozeroy", fillcolor="rgba(5,150,105,0.07)"))
                fig_ot.update_layout(title="OTD — 납기준수율 (%)", height=220,
                                      margin=dict(l=40,r=20,t=40,b=40),
                                      plot_bgcolor="white", paper_bgcolor="white",
                                      yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="#f1f5f9"))
                st.plotly_chart(fig_ot, use_container_width=True)

        # Station state summary
        stn_states = cur.get("stations", [])
        if stn_states:
            st.markdown('<div class="sec-header">🔧 설비 상태 분포</div>', unsafe_allow_html=True)
            state_counts = {"busy":0,"idle":0,"down":0,"setup":0}
            for s in stn_states:
                state_counts[s.get("state","idle")] = state_counts.get(s.get("state","idle"),0) + 1
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("🔵 Busy",  state_counts["busy"])
            col2.metric("⬛ Idle",  state_counts["idle"])
            col3.metric("🔴 Down",  state_counts["down"])
            col4.metric("🟡 Setup", state_counts["setup"])

        # GNN action log
        gnn_logs = cur.get("gnn_logs", [])
        if gnn_logs:
            st.markdown('<div class="sec-header">🤖 GNN + XGBoost Action Log</div>', unsafe_allow_html=True)
            log_html = '<div style="background:#0f172a;border-radius:10px;padding:14px;font-family:JetBrains Mono,monospace;font-size:11px;max-height:200px;overflow-y:auto;">'
            for log in gnn_logs[-20:]:
                color = "#ef4444" if "CRITICAL" in log or "Warning" in log else "#22c55e" if "Resolved" in log else "#60a5fa" if "GNN" in log else "#fbbf24"
                log_html += f'<div style="color:{color};margin-bottom:3px;">▸ {log}</div>'
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 3: BENCHMARK ARENA
    # ════════════════════════════════════════════════════════
    with tab_bench:
        st.markdown('<div class="sec-header">🏆 Benchmark Arena — Dispatching Policy Comparison</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-blue" style="margin-bottom:14px;">
        💡 사이드바에서 <b>Dispatching Policy</b>를 바꾸고 <b>🚀 정책 적용</b>을 누른 뒤 시뮬레이션을 진행하면 
        결과가 자동으로 아래 표에 누적됩니다.
        </div>
        """, unsafe_allow_html=True)

        # DS4 reference results (from actual simulation)
        ref_results = {
            "FIFO":    {"ct":1071, "ontime":43.15, "cqt_viol":1451, "hotlot_otd":25.86},
            "EDD":     {"ct":909,  "ontime":54.76, "cqt_viol":1390, "hotlot_otd":68.67},
            "Oracle":  {"ct":921,  "ontime":66.74, "cqt_viol":1255, "hotlot_otd":64.91},
            "GNN+XGB": {"ct":748,  "ontime":81.16, "cqt_viol":880,  "hotlot_otd":91.30},
        }

        st.markdown("**DS4 최종 실험 결과 (확정 수치)**")
        rows = []
        for pol, r in ref_results.items():
            vs_fifo_ct = round((r["ct"]-1071)/1071*100, 1)
            vs_fifo_ot = round(r["ontime"] - 43.15, 1)
            rows.append({
                "Policy": f"★ {pol}" if pol == "GNN+XGB" else pol,
                "OTD (납기준수율)": f"{r['ontime']}%",
                "Avg CT (평균 공정시간)": f"{r['ct']}h",
                "CQT 위반": f"{r['cqt_viol']:,}건",
                "HotLot OTD": f"{r['hotlot_otd']}%",
                "CT vs FIFO": f"{vs_fifo_ct:+.1f}%",
                "OTD vs FIFO": f"{vs_fifo_ot:+.1f}%p",
            })
        df_ref = pd.DataFrame(rows)

        def highlight_best(s):
            styles = []
            for val in s:
                if "★" in str(val) or "GNN" in str(val):
                    styles.append("background-color:#dbeafe;font-weight:700;color:#1e40af")
                else:
                    styles.append("")
            return styles

        st.dataframe(
            df_ref.style.apply(highlight_best, axis=0, subset=["Policy"]),
            use_container_width=True, hide_index=True
        )

        # Summary metrics
        st.markdown("**성능 개선 요약 (vs FIFO 기준선)**")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("OTD 개선", "+38.0%p", "GNN+XGB vs FIFO")
        m2.metric("CT 단축", "-30.2%", "748h → FIFO 1,071h")
        m3.metric("CQT 위반 감소", "-39.4%", "880건 → FIFO 1,451건")
        m4.metric("HotLot OTD", "+65.4%p", "91.30% → FIFO 25.86%")

        # Live benchmark from current simulation
        if st.session_state.benchmark_data:
            st.markdown("---")
            st.markdown("**현재 세션 Live Benchmark**")
            live_rows = []
            for pol, d in st.session_state.benchmark_data.items():
                live_rows.append({
                    "Policy": pol,
                    "Avg CT (h)": f"{d.get('ct',0):.1f}",
                    "OTD (%)": f"{d.get('ontime',0):.1f}",
                    "WIP": f"{d.get('wip',0):,}",
                })
            st.dataframe(pd.DataFrame(live_rows), use_container_width=True, hide_index=True)

        # Breakdown risk table
        st.markdown('<div class="sec-header">⚙️ 설비 고장 리스크 — MTTF / MTTR</div>', unsafe_allow_html=True)
        bd_data = [
            {"Area":"Litho","MTTF (min)":10080,"MTTR (min)":705.59,"Availability":"93.5%","Risk":"🔴 HIGH"},
            {"Area":"Implant","MTTF (min)":10080,"MTTR (min)":604.8,"Availability":"94.3%","Risk":"🔴 HIGH"},
            {"Area":"Dielectric","MTTF (min)":10080,"MTTR (min)":604.8,"Availability":"94.3%","Risk":"🔴 HIGH"},
            {"Area":"TF","MTTF (min)":10080,"MTTR (min)":453.6,"Availability":"95.7%","Risk":"🟡 MED"},
            {"Area":"Planar","MTTF (min)":10080,"MTTR (min)":201.6,"Availability":"98.1%","Risk":"🟡 MED"},
            {"Area":"Dry_Etch","MTTF (min)":10080,"MTTR (min)":231.84,"Availability":"97.8%","Risk":"🟡 MED"},
            {"Area":"Wet_Etch","MTTF (min)":10080,"MTTR (min)":221.76,"Availability":"97.9%","Risk":"🟡 MED"},
            {"Area":"Diffusion","MTTF (min)":10080,"MTTR (min)":151.2,"Availability":"98.5%","Risk":"🟢 LOW"},
            {"Area":"Def_Met","MTTF (min)":10080,"MTTR (min)":35.28,"Availability":"99.7%","Risk":"🟢 LOW"},
            {"Area":"Litho_Met","MTTF (min)":10080,"MTTR (min)":35.28,"Availability":"99.7%","Risk":"🟢 LOW"},
            {"Area":"TF_Met","MTTF (min)":10080,"MTTR (min)":35.28,"Availability":"99.7%","Risk":"🟢 LOW"},
        ]
        st.dataframe(pd.DataFrame(bd_data), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # TAB 4: 재난 시뮬레이터
    # ════════════════════════════════════════════════════════
    with tab_disaster:
        st.markdown('<div class="sec-header">🚨 재난 시뮬레이터 — What-if Scenario Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-amber" style="margin-bottom:14px;">
        ⚡ <b>Resilience Score 산출 도구</b> — 교란 시나리오 발생 시 KPI 회복 속도를 정량화합니다.
        대기업 발주처 실사 시 납기 신뢰도를 수치로 증명하는 유일한 도구입니다.
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.markdown("**시나리오 선택**")
            scenario_name = st.selectbox("재난 시나리오", list(DISASTER_SCENARIOS.keys()))
            scenario = DISASTER_SCENARIOS[scenario_name]
            st.markdown(f'<div class="alert-red" style="margin:10px 0;">{scenario["desc"]}</div>', unsafe_allow_html=True)

            st.markdown("**추가 파라미터**")
            extra_hotlot = st.slider("긴급 HotLot 추가 투입 수", 0, 100, 0, 10)
            wip_reduction = st.slider("WIP_CAP 즉시 감소율 (%)", 0, 50, 20, 5,
                                       help="Floodgate — 수문 닫힘 강도")
            duration_mult = st.slider("장애 지속 시간 배율", 0.5, 3.0, 1.0, 0.5)

            if st.button("🚨 재난 시나리오 실행", type="primary", use_container_width=True):
                # Simulate disaster effect
                actual_duration = int(scenario["duration"] * duration_mult)
                base_ct = BASELINE_MAP[st.session_state.ds_id]

                # Heuristic impact model
                area = scenario["area"]
                mttr_map = {"Litho": 705.59, "Diffusion": 151.2, "Implant": 604.8}
                mttr = mttr_map.get(area, 200)
                impact_factor = min(actual_duration / (base_ct * 60) * 2.5, 0.8)

                degraded_ct   = round(base_ct * (1 + impact_factor * 0.6))
                degraded_otd  = max(0, round(43.15 - impact_factor * 40))
                degraded_cqt  = round(1451 * (1 + impact_factor))
                recovery_time = round(actual_duration * 0.3 + mttr * 2)

                # Resilience Score (0-100)
                resilience = max(0, min(100, round(100 - impact_factor * 60 - extra_hotlot * 0.3 + wip_reduction * 0.5)))

                st.session_state.disaster_log.append({
                    "scenario": scenario_name,
                    "duration": actual_duration,
                    "extra_hotlot": extra_hotlot,
                    "wip_reduction": wip_reduction,
                    "degraded_ct": degraded_ct,
                    "degraded_otd": degraded_otd,
                    "degraded_cqt": degraded_cqt,
                    "recovery_time": recovery_time,
                    "resilience": resilience,
                })
                st.rerun()

        with col_right:
            if st.session_state.disaster_log:
                last = st.session_state.disaster_log[-1]
                st.markdown("**최근 시나리오 결과**")

                res_color = "#22c55e" if last["resilience"] >= 70 else "#f59e0b" if last["resilience"] >= 40 else "#ef4444"
                st.markdown(f"""
                <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px;">
                    <div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:12px;">
                        📋 {last['scenario']}
                    </div>
                    <div style="display:flex;justify-content:center;margin-bottom:16px;">
                        <div style="text-align:center;">
                            <div style="font-size:52px;font-weight:700;color:{res_color};">{last['resilience']}</div>
                            <div style="font-size:13px;color:#64748b;">Resilience Score</div>
                        </div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                        <div style="background:#fef2f2;border-radius:8px;padding:10px;text-align:center;">
                            <div style="font-size:10px;color:#ef4444;font-weight:600;">CT 영향</div>
                            <div style="font-size:18px;font-weight:700;color:#ef4444;">{last['degraded_ct']}h</div>
                            <div style="font-size:10px;color:#94a3b8;">정상 {BASELINE_MAP[st.session_state.ds_id]}h</div>
                        </div>
                        <div style="background:#fef2f2;border-radius:8px;padding:10px;text-align:center;">
                            <div style="font-size:10px;color:#ef4444;font-weight:600;">OTD 영향</div>
                            <div style="font-size:18px;font-weight:700;color:#ef4444;">{last['degraded_otd']}%</div>
                            <div style="font-size:10px;color:#94a3b8;">정상 43.15%</div>
                        </div>
                        <div style="background:#fffbeb;border-radius:8px;padding:10px;text-align:center;">
                            <div style="font-size:10px;color:#f59e0b;font-weight:600;">CQT 위반</div>
                            <div style="font-size:18px;font-weight:700;color:#f59e0b;">{last['degraded_cqt']:,}건</div>
                        </div>
                        <div style="background:#eff6ff;border-radius:8px;padding:10px;text-align:center;">
                            <div style="font-size:10px;color:#2563eb;font-weight:600;">Recovery Time</div>
                            <div style="font-size:18px;font-weight:700;color:#2563eb;">{last['recovery_time']}min</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if last["resilience"] < 50:
                    st.markdown('<div class="alert-red" style="margin-top:10px;">⚠️ Floodgate 자동 발동 — WIP_CAP 즉시 감소 권장</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-green" style="margin-top:10px;">✅ 공장 회복력 양호 — 납기 신뢰도 유지 가능</div>', unsafe_allow_html=True)

        # Scenario history
        if len(st.session_state.disaster_log) > 1:
            st.markdown("---")
            st.markdown("**시나리오 비교 이력**")
            df_dis = pd.DataFrame(st.session_state.disaster_log)[
                ["scenario","duration","degraded_ct","degraded_otd","degraded_cqt","resilience"]
            ]
            df_dis.columns = ["시나리오","장애시간(min)","CT(h)","OTD(%)","CQT위반","Resilience"]
            st.dataframe(df_dis, use_container_width=True, hide_index=True)

        # TOC Buffer concept
        st.markdown('<div class="sec-header">🔩 Floodgate Control — Conditional WIP 상한</div>', unsafe_allow_html=True)
        st.markdown("""
        | 버퍼 상태 | 조건 | Floodgate 조치 | TOC 대응 |
        |-----------|------|----------------|----------|
        | 🟢 Green (정상) | Down = 0 | WIP_CAP 원래 값 유지 | 정상 투입 |
        | 🟡 Yellow (주의) | Down = 1 | WIP_CAP 5% 감소 | 투입 소폭 조절 |
        | 🔴 Red (위험) | Down ≥ 2 | WIP_CAP 20% 감소 | 투입 즉시 억제 |
        """)

    # ════════════════════════════════════════════════════════
    # TAB 5: AI EXPLAINABILITY
    # ════════════════════════════════════════════════════════
    with tab_xai:
        st.markdown('<div class="sec-header">🔍 AI Explainability — GNN + XGBoost</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-blue" style="margin-bottom:14px;">
        🧠 A-APOS의 핵심 차별점: RL 블랙박스와 달리 <b>모든 의사결정에 대한 설명</b>이 가능합니다.
        GNN Attention 가중치 + XGBoost SHAP으로 "왜 이 Lot을 먼저 처리했는가"를 추적합니다.
        </div>
        """, unsafe_allow_html=True)

        col_gnn, col_xgb = st.columns(2)

        with col_gnn:
            st.markdown("**GNN Dispatcher — DS별 피처 중요도 전략**")
            ds_feat_data = {
                "DS": ["DS1 (단순)","DS2 (다품종)","DS3 (고장)","DS4 (최악)"],
                "지배적 피처": ["CR, waiting_time, priority","CR, remaining_steps, station_queue",
                               "area_down_rate, cqt_urgency, cascade_risk","All 13 features"],
                "모델 전략": ["기본 4개 피처","중간 7개 피처","고장 피처 추가","전체 통합"],
                "핵심 이유": ["고장 없음 → 납기·대기 시간 지배","다품종 혼잡 → 큐 상태 중요",
                              "고장 발생 → 연쇄 피처 필수","모든 변수 통합 필요"],
            }
            st.dataframe(pd.DataFrame(ds_feat_data), use_container_width=True, hide_index=True)

            st.markdown("**CQT Urgency — EWS (Early Warning Score) 메커니즘**")
            st.markdown("""
            ```python
            CQT_urgency = CQT잔여시간 / 해당설비_평균처리시간
            
            if urgency < 2.0:  → CRITICAL  (GNN score +100)
            if urgency < 5.0:  → URGENT    (GNN score +50)
            
            # Priority 10 lot vs Priority 50 lot 경합 시:
            # urgency < 2.0 → Priority 50 lot 순위 즉시 역전
            ```
            """)

        with col_xgb:
            st.markdown("**XGBoost — 83개 피처 기여도 (DS별 예상 SHAP Top-3)**")
            shap_data = {
                "DS": ["DS1","DS2","DS3","DS4"],
                "SHAP #1": ["critical_ratio","station_queue","area_down_rate","cascade_risk"],
                "SHAP #2": ["waiting_time","remaining_steps","cqt_urgency","WIP_slope"],
                "SHAP #3": ["priority","cqt_urgency","cascade_risk","cqt_urgency"],
                "핵심 인사이트": [
                    "납기·대기 지배",
                    "다품종 큐 혼잡",
                    "고장이 모든 것",
                    "연쇄·추세가 결정",
                ],
            }
            st.dataframe(pd.DataFrame(shap_data), use_container_width=True, hide_index=True)

            st.markdown("**98% 포화 문제 해결 — Sigmoid 적용**")
            # Simple visualization of sigmoid vs linear
            import plotly.graph_objects as go
            x_vals = list(range(0, 101, 5))
            linear = [min(x, 98) for x in x_vals]
            sigmoid = [round(1/(1+2.718**(-0.1*(x-50)))*95+2, 1) for x in x_vals]

            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=x_vals, y=linear, name="기존 Linear (포화)",
                                          line=dict(color="#ef4444", dash="dash", width=1.5)))
            fig_sig.add_trace(go.Scatter(x=x_vals, y=sigmoid, name="Sigmoid (해결)",
                                          line=dict(color="#2563eb", width=2)))
            fig_sig.update_layout(
                title="Sigmoid vs Linear 확률 변환", height=220,
                margin=dict(l=40,r=20,t=40,b=40),
                xaxis_title="Raw Score", yaxis_title="Prob (%)",
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(gridcolor="#f1f5f9", range=[0,105]),
                xaxis=dict(gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig_sig, use_container_width=True)

        # Floodgate log
        st.markdown('<div class="sec-header">🚧 Floodgate Control Log</div>', unsafe_allow_html=True)
        gate_logs = []
        if "bridge" in st.session_state:
            try:
                gate_logs = st.session_state.bridge.gate_logs
            except:
                pass
        if gate_logs:
            for log in gate_logs[-10:]:
                color = "#ef4444" if "Active" in log else "#22c55e"
                st.markdown(f'<div style="font-size:12px;color:{color};padding:3px 0;font-family:monospace;">{log}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-green">✅ Floodgate 정상 — 현재 모든 구역 고장 없음</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# LLM PANEL (right column)
# ════════════════════════════════════════════════════════
with llm_col:
    st.markdown("""
    <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                padding:14px 16px;height:100%;min-height:600px;">
        <div style="font-size:13px;font-weight:600;color:#0f172a;border-bottom:1px solid #e2e8f0;
                    padding-bottom:10px;margin-bottom:12px;">
            🤖 A-APOS Assistant
        </div>
    """, unsafe_allow_html=True)

    # API key input
    api_key = st.text_input("OpenAI API Key", type="password",
                             placeholder="sk-...",
                             label_visibility="collapsed")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-ai">
            안녕하세요! 저는 A-APOS 대시보드 어시스턴트입니다.<br><br>
            다음과 같은 질문을 할 수 있습니다:<br>
            • "Litho 구역 병목의 원인은?"<br>
            • "현재 OTD가 낮은 이유는?"<br>
            • "Floodgate를 언제 발동해야 하나?"<br>
            • "GNN이 FIFO보다 나은 이유는?"
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history[-8:]:
                css_class = "chat-user" if msg["role"] == "user" else "chat-ai"
                st.markdown(f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)

    # Quick prompts
    st.markdown("**빠른 질문**")
    quick_qs = [
        "Litho 병목 원인",
        "GNN vs FIFO 차이",
        "재난 발생 시 대응",
        "CQT urgency 설명",
    ]
    for q in quick_qs:
        if st.button(q, key=f"qq_{q}", use_container_width=True):
            st.session_state.chat_history.append({"role":"user","content":q})
            if api_key:
                try:
                    import openai
                    client = openai.OpenAI(api_key=api_key)

                    # Context from current state
                    ctx_kpi = cur.get("kpi", {})
                    ctx_xgb = {k: round(v*100) for k,v in cur.get("xgb_probs",{}).items() if v > 0.3}
                    system_prompt = f"""당신은 SMT 반도체 FAB A-APOS 시스템의 전문 어시스턴트입니다.
현재 상태: WIP={cur.get('wip',0)}, OTD={ctx_kpi.get('ontime_pct',0)}%, CT={ctx_kpi.get('avg_ct',0)}h, 
CQT위반={cur.get('cqt',{}).get('violations',0)}건
XGBoost 병목 예측: {ctx_xgb}
Dataset: {DS_LABELS[st.session_state.ds_id]}
Policy: {st.session_state.overrides.get('policy','GNN')}
현재 상태를 바탕으로 실무적이고 간결하게 답변하세요. 200자 이내로."""

                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":system_prompt}] +
                                  st.session_state.chat_history[-6:],
                        max_tokens=300,
                        temperature=0.3,
                    )
                    answer = resp.choices[0].message.content
                except Exception as e:
                    answer = f"API 오류: {str(e)[:80]}. API 키를 확인해주세요."
            else:
                # Fallback answers without API
                fallback = {
                    "Litho 병목 원인": "Litho 구역은 MTTR이 705.59분으로 가장 높습니다. 설비 고장 시 WIP이 폭발적으로 증가하며, Cascade_risk 피처를 통해 Implant 구역으로 연쇄 전파됩니다. Floodgate 발동 권장.",
                    "GNN vs FIFO 차이": "FIFO는 현재 대기열만 보지만, GNN은 구역→구역 엣지를 통해 미래 경합 경로를 예측합니다. DS4 실험 결과 OTD +38%p, CT -30% 달성.",
                    "재난 발생 시 대응": "1) Floodgate 자동 발동 (WIP_CAP 20% 감소) 2) XGBoost 경고 확인 3) What-if 시뮬레이션으로 시나리오 비교 4) Resilience Score 산출.",
                    "CQT urgency 설명": "CQT_urgency = CQT잔여시간 / 설비평균처리시간. 2.0 미만이면 CRITICAL (+100점), 5.0 미만이면 URGENT (+50점). 의료 ICU EWS 개념 이식.",
                }
                answer = fallback.get(q, "API 키를 입력하면 현재 시뮬레이션 상태를 기반으로 답변합니다.")
            st.session_state.chat_history.append({"role":"assistant","content":answer})
            st.rerun()

    # Free input
    user_input = st.text_input("질문 입력", placeholder="공장 상태에 대해 질문하세요...",
                                label_visibility="collapsed", key="chat_input")
    if st.button("전송", use_container_width=True) and user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        if api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                ctx_kpi = cur.get("kpi", {})
                system_prompt = f"""당신은 SMT 반도체 FAB A-APOS 전문 어시스턴트입니다.
현재 WIP={cur.get('wip',0)}, OTD={ctx_kpi.get('ontime_pct',0)}%, CT={ctx_kpi.get('avg_ct',0)}h.
실무적이고 간결하게 200자 이내로 답변하세요."""
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":system_prompt}] +
                              st.session_state.chat_history[-6:],
                    max_tokens=300, temperature=0.3,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"오류: {str(e)[:80]}"
        else:
            answer = "API 키를 상단에 입력하면 현재 시뮬레이션 상태를 반영한 답변을 드립니다."
        st.session_state.chat_history.append({"role":"assistant","content":answer})
        st.rerun()

    if st.button("대화 초기화", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ── Simulation loop ───────────────────────────────────────────────────────────
if st.session_state.running and ENGINE_OK and "bridge" in st.session_state:
    xgb_probs = cur.get("xgb_probs", {})
    high_prob = [a for a, p in xgb_probs.items() if p > 0.7]
    if high_prob:
        st.session_state.running = False
        st.warning(f"🚨 Auto-Pause: {', '.join(high_prob)} 구역 병목 확률 > 70%")
        st.rerun()

    st.session_state.tick += sim_speed
    try:
        st.session_state.bridge.run_step(until=st.session_state.tick)
    except Exception as e:
        st.error(f"시뮬레이션 오류: {e}")
        st.session_state.running = False
    time.sleep(0.05)
    st.rerun()
