"""
A-APOS v3.0 — app.py (Final)
- 데이터 경로 수정 (새 레포 구조 기준)
- dashboard.html 렌더링 복원
- 멀티탭 추가
- LLM 패널 추가
- 사이드바 글씨 대비 수정 (시작 시 잘 보이게)
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

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(BASE_DIR, "A_APOS_Engine")

# 새 레포 구조: app.py와 SMT_2020 폴더가 같은 레벨
DATA_PATH = os.path.join(BASE_DIR, "SMT_2020 - Final", "AutoSched")

BASELINE_MAP = {1: 949, 2: 897, 3: 923, 4: 955}
DS_LABELS = {
    1: "DS1 · HVLM (소품종 대량, 고장 없음)",
    2: "DS2 · LVHM (다품종 소량, 고장 없음)",
    3: "DS3 · HVLM_E (소품종, 고장 포함)",
    4: "DS4 · LVHM_E (다품종, 고장 포함)",
}
DISASTER_SCENARIOS = {
    "A — Litho 48h 전면 다운":           {"area": "Litho",     "duration": 2880},
    "B — Diffusion 가스 공급 중단 72h":  {"area": "Diffusion", "duration": 4320},
    "C — 복합 재난 (Litho + HotLot 50)": {"area": "Litho",     "duration": 2880},
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #f8fafc; }

/* ── 사이드바: running 여부에 따라 텍스트 색상 분기 ── */
[data-testid="stSidebar"] { background: #0f172a !important; border-right: 1px solid #1e293b; }
/* 기본: 중단 상태 → 흐린 색 */
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
/* metric value는 항상 밝게 */
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 20px !important; font-weight: 700 !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; }
/* 버튼 */
[data-testid="stSidebar"] .stButton > button { color: #e2e8f0 !important; background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 8px; font-weight: 500; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #2563eb !important; border-color: #2563eb !important; color: white !important; }
/* selectbox, slider label */
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 12px !important; }
/* selectbox 선택된 값 */
[data-testid="stSidebar"] [data-baseweb="select"] { background: #1e293b !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #e2e8f0 !important; }
/* number input */
[data-testid="stSidebar"] input { color: #e2e8f0 !important; background: #1e293b !important; }
/* divider */
[data-testid="stSidebar"] hr { border-color: #1e293b !important; }
/* caption */
[data-testid="stSidebar"] .stCaption { color: #475569 !important; }

/* ── 탭 ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] { background: white; border-radius: 10px; padding: 4px; border: 1px solid #e2e8f0; gap: 2px; }
[data-testid="stTabs"] [data-baseweb="tab"] { border-radius: 8px; font-weight: 500; font-size: 13px; color: #64748b; padding: 8px 18px; }
[data-testid="stTabs"] [aria-selected="true"] { background: #2563eb !important; color: white !important; }

/* ── Metric ── */
[data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
[data-testid="stMetricLabel"] { font-size: 11px !important; color: #64748b !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700 !important; color: #0f172a !important; }

/* ── 버튼 ── */
.stButton > button { border-radius: 8px; font-weight: 500; border: 1px solid #e2e8f0; transition: all 0.15s; }
.stButton > button[kind="primary"] { background: #2563eb; border-color: #2563eb; color: white; }

/* ── Alert boxes ── */
.alert-red   { background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:10px 14px; color:#991b1b; font-size:13px; margin:6px 0; }
.alert-amber { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:10px 14px; color:#92400e; font-size:13px; margin:6px 0; }
.alert-green { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:10px 14px; color:#14532d; font-size:13px; margin:6px 0; }
.alert-blue  { background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:10px 14px; color:#1e40af; font-size:13px; margin:6px 0; }

/* ── LLM chat ── */
.chat-user { background:#2563eb; color:white; border-radius:12px 12px 2px 12px; padding:10px 14px; margin:6px 0 6px 40px; font-size:13px; }
.chat-ai   { background:white; color:#1e293b; border:1px solid #e2e8f0; border-radius:12px 12px 12px 2px; padding:10px 14px; margin:6px 40px 6px 0; font-size:13px; line-height:1.6; }

/* ── 섹션 헤더 ── */
.sec-header { font-size:13px; font-weight:600; color:#374151; margin:16px 0 10px; padding-bottom:6px; border-bottom:2px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Engine import ─────────────────────────────────────────────────────────────
try:
    from A_APOS_Engine.data_manager import APOSDataManager
    from A_APOS_Engine.engine_wrapper import SimBridge
    ENGINE_OK = True
except Exception as e:
    ENGINE_OK = False
    ENGINE_ERR = str(e)

# ── Session State 초기화 ──────────────────────────────────────────────────────
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

for k, v in [
    ("ds_id", 4), ("tick", 0), ("running", False),
    ("kpi_log", []), ("gnn_log_history", []),
    ("benchmark_data", {}), ("disaster_log", []),
    ("chat_history", []), ("overrides", {}), ("last_kpi", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

if "bridge" not in st.session_state and ENGINE_OK:
    init_session(4)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    running = st.session_state.get("running", False)

    st.markdown(f"""
    <div style="padding:16px 0 12px; border-bottom:1px solid #1e293b; margin-bottom:16px;">
        <div style="font-size:20px; font-weight:700; color:#f1f5f9;">🏭 A-APOS</div>
        <div style="font-size:11px; color:#475569; margin-top:2px;">Factory OS v3.0 · SMT 2020</div>
        <div style="margin-top:8px; display:inline-block; padding:3px 10px; border-radius:20px;
             background:{'#16a34a' if running else '#374151'}; font-size:11px;
             color:{'#dcfce7' if running else '#94a3b8'}; font-weight:600;">
            {'▶ RUNNING' if running else '⏹ STOPPED'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset 선택
    st.markdown('<div style="font-size:11px;color:#64748b;font-weight:600;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.06em;">📂 Dataset</div>', unsafe_allow_html=True)
    ds_choice = st.selectbox("", [1,2,3,4], index=st.session_state.ds_id-1,
                              format_func=lambda x: DS_LABELS[x], label_visibility="collapsed")
    if ds_choice != st.session_state.ds_id:
        init_session(ds_choice)
        st.rerun()

    st.markdown("---")

    # Simulation control
    st.markdown('<div style="font-size:11px;color:#64748b;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em;">⚙️ Simulation Control</div>', unsafe_allow_html=True)
    sim_speed = st.slider("Step Size (분)", 10, 500, 50, step=10)
    policy = st.selectbox("Dispatching Policy", ["GNN","FIFO","EDD","CR"],
                           index=["GNN","FIFO","EDD","CR"].index(st.session_state.overrides.get("policy","GNN")))
    wip_limit = st.number_input("WIP Limit", 1000, 10000,
                                 st.session_state.overrides.get("wip_limit", 3000), step=500)
    cap_factor  = st.slider("Capacity 배율", 0.5, 2.0, st.session_state.overrides.get("capacity_factor", 1.0), 0.1)
    mttf_factor = st.slider("MTTF 배율", 0.5, 3.0, st.session_state.overrides.get("mttf_factor", 1.0), 0.1)
    mttr_factor = st.slider("MTTR 배율", 0.5, 3.0, st.session_state.overrides.get("mttr_factor", 1.0), 0.1)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ 시작", use_container_width=True, type="primary"):
            st.session_state.running = True
    with c2:
        if st.button("⏹ 중단", use_container_width=True):
            st.session_state.running = False

    c3, c4 = st.columns(2)
    with c3:
        if st.button("🔄 초기화", use_container_width=True):
            init_session(st.session_state.ds_id, {"policy": policy, "wip_limit": wip_limit,
                                                    "capacity_factor": cap_factor,
                                                    "mttf_factor": mttf_factor, "mttr_factor": mttr_factor})
            st.rerun()
    with c4:
        if st.button("🚀 정책 적용", use_container_width=True):
            if "bridge" in st.session_state:
                st.session_state.last_kpi = {
                    "policy": st.session_state.overrides.get("policy","GNN"),
                    "ct": st.session_state.bridge.kpi_history[-1]["ct"] if st.session_state.bridge.kpi_history else 0,
                    "ontime": st.session_state.bridge.kpi_history[-1]["ontime"] if st.session_state.bridge.kpi_history else 0,
                }
            init_session(st.session_state.ds_id, {"policy": policy, "wip_limit": wip_limit,
                                                    "capacity_factor": cap_factor,
                                                    "mttf_factor": mttf_factor, "mttr_factor": mttr_factor})
            st.rerun()

    st.markdown("---")

    # Live KPI — 시작 상태일 때 밝게 표시
    st.markdown(f'<div style="font-size:11px;color:#64748b;font-weight:600;margin-bottom:8px;text-transform:uppercase;">📊 Live KPI</div>', unsafe_allow_html=True)
    st.metric("Sim Time", f"T+{st.session_state.tick}")

    if ENGINE_OK and "bridge" in st.session_state:
        try:
            s = st.session_state.bridge.get_summary()
            kh = st.session_state.bridge.kpi_history
            st.metric("WIP (재공품)", f"{s['wip']:,}")
            st.metric("완료 Lot", f"{s['completed']:,}")
            if kh:
                st.metric("Avg CT", f"{kh[-1]['ct']:.0f} h")
                st.metric("OTD", f"{kh[-1]['ontime']:.1f} %")
            else:
                st.metric("Avg CT", "0 h")
                st.metric("OTD", "0.0 %")
            c1, c2, c3 = st.columns(3)
            c1.metric("Busy", s["busy"])
            c2.metric("Down", s["down"])
            c3.metric("Idle", s["idle"])
        except:
            pass

    # Scenario comparison
    if st.session_state.last_kpi:
        st.markdown("---")
        st.markdown(f'<div style="font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;">🏁 Scenario Comparison</div>', unsafe_allow_html=True)
        if "bridge" in st.session_state:
            kh = st.session_state.bridge.kpi_history
            if kh:
                curr = kh[-1]
                last = st.session_state.last_kpi
                ct_diff = curr["ct"] - last.get("ct", 0)
                ot_diff = curr["ontime"] - last.get("ontime", 0)
                st.metric("CT 변화", f"{curr['ct']:.1f} h", f"{ct_diff:+.1f} h", delta_color="inverse")
                st.metric("OTD 변화", f"{curr['ontime']:.1f} %", f"{ot_diff:+.1f} %")

    st.markdown("---")
    st.caption(f"Baseline LT: {BASELINE_MAP[st.session_state.ds_id]}h")

# ── Get current state ─────────────────────────────────────────────────────────
cur = {}
if ENGINE_OK and "bridge" in st.session_state:
    try:
        cur = st.session_state.bridge.update_ui_state()
        # GNN 로그 누적
        if "gnn_logs" in cur:
            st.session_state.gnn_log_history.extend(cur["gnn_logs"])
            st.session_state.gnn_log_history = st.session_state.gnn_log_history[-60:]
        cur["gnn_logs"] = list(st.session_state.gnn_log_history)

        # Benchmark 누적
        kh = st.session_state.bridge.kpi_history
        if kh:
            pol = st.session_state.overrides.get("policy","GNN")
            st.session_state.benchmark_data[pol] = {
                "ct": kh[-1]["ct"], "ontime": kh[-1]["ontime"],
                "wip": cur.get("wip", 0), "cqt": cur.get("cqt",{}).get("violations",0),
            }
        # KPI 로그
        st.session_state.kpi_log.append({
            "tick": st.session_state.tick, "wip": cur.get("wip",0),
            "ct": cur.get("kpi",{}).get("avg_ct",0),
            "ontime": cur.get("kpi",{}).get("ontime_pct",0),
            "down": cur.get("kpi",{}).get("down_count",0),
        })
    except Exception as e:
        st.error(f"상태 업데이트 오류: {e}")

# ── Main layout ───────────────────────────────────────────────────────────────
main_col, llm_col = st.columns([3.2, 0.8])

with main_col:
    # Header
    kpi = cur.get("kpi", {})
    kh_data = st.session_state.get("bridge") and "bridge" in st.session_state and st.session_state.bridge.kpi_history or []
    last_kh = kh_data[-1] if kh_data else {}
    running = st.session_state.get("running", False)

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                background:white;border-radius:12px;padding:14px 20px;
                border:1px solid #e2e8f0;margin-bottom:12px;
                box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <div>
            <div style="font-size:20px;font-weight:700;color:#0f172a;">A-APOS Factory Operating System</div>
            <div style="font-size:12px;color:#64748b;margin-top:2px;">
                {DS_LABELS[st.session_state.ds_id]} &nbsp;|&nbsp;
                Policy: <b style="color:#2563eb;">{st.session_state.overrides.get('policy','GNN')}</b> &nbsp;|&nbsp;
                T+{st.session_state.tick} &nbsp;|&nbsp; Baseline {BASELINE_MAP[st.session_state.ds_id]}h
            </div>
        </div>
        <div style="text-align:right;">
            <div style="display:inline-block;padding:5px 14px;border-radius:20px;
                 background:{'#dcfce7' if running else '#f1f5f9'};
                 color:{'#15803d' if running else '#64748b'};
                 font-size:12px;font-weight:700;">
                {'▶ SIM RUNNING' if running else '⏹ STOPPED'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab_main, tab_dashboard, tab_bench, tab_disaster, tab_xai = st.tabs([
        "🏠  Overview",
        "📡  Live Dashboard",
        "🏆  Benchmark Arena",
        "🚨  재난 시뮬레이터",
        "🔍  AI Explainability",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ════════════════════════════════════════════════════════
    with tab_main:
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("WIP (재공품)", f"{cur.get('wip',0):,}")
        with c2: st.metric("완료 Lot", f"{kpi.get('completed',0):,}")
        with c3:
            ct_val = last_kh.get('ct', 0)
            delta_ct = ct_val - BASELINE_MAP[st.session_state.ds_id]
            st.metric("Avg Cycle Time", f"{ct_val:.0f} h",
                      delta=f"{delta_ct:+.0f}h vs baseline", delta_color="inverse")
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
            util  = stats.get("avg_util", round(stats.get("busy",0)/total*100,1))
            down  = stats.get("down", 0)
            wip   = stats.get("wip", 0)
            prob  = xgb_probs.get(a["name"], 0)
            avail = round(a["mttf"]/(a["mttf"]+a["mttr"])*100, 1)
            if prob >= 0.7 or down >= 2:
                bc, bg, ic = "#ef4444", "#fef2f2", "🔴"
            elif prob >= 0.4 or down >= 1:
                bc, bg, ic = "#f59e0b", "#fffbeb", "🟡"
            else:
                bc, bg, ic = "#22c55e", "#f0fdf4", "🟢"
            bar = f'<div style="background:#e2e8f0;border-radius:3px;height:5px;margin:6px 0;"><div style="width:{min(util,100)}%;background:{bc};height:5px;border-radius:3px;"></div></div>'
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {bc};border-radius:10px;
                            padding:12px;margin-bottom:8px;border-left:4px solid {bc};">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-size:12px;font-weight:600;color:#1e293b;">{ic} {a['name']}</div>
                        <div style="font-size:11px;font-weight:700;color:{bc};">{prob*100:.0f}%</div>
                    </div>
                    {bar}
                    <div style="display:flex;justify-content:space-between;font-size:10px;color:#64748b;">
                        <span>Util {util:.0f}%</span><span>WIP {wip}</span>
                        <span style="color:{'#ef4444' if down>0 else '#94a3b8'}">Down {down}</span>
                    </div>
                    <div style="font-size:9px;color:#94a3b8;margin-top:4px;">Avail {avail}% · MTTR {a['mttr']}min</div>
                </div>
                """, unsafe_allow_html=True)

        high_prob = [(a, round(xgb_probs[a]*100)) for a in xgb_probs if xgb_probs[a] >= 0.7]
        if high_prob:
            w = ", ".join([f"<b>{a}</b> ({p}%)" for a,p in high_prob])
            st.markdown(f'<div class="alert-red">🚨 <b>Bottleneck Alert</b> — XGBoost 예측: {w} 구역 위험 (≥70%)</div>', unsafe_allow_html=True)

        cqt = cur.get("cqt", {})
        urgent_list = cqt.get("urgent_list", [])
        if urgent_list:
            st.markdown('<div class="sec-header">⚠️ CQT 긴급 Lot 현황</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(urgent_list), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # TAB 2: LIVE DASHBOARD (기존 dashboard.html 렌더링)
    # ════════════════════════════════════════════════════════
    with tab_dashboard:
        st.markdown('<div class="sec-header">📡 Live Brain Dashboard (Real-time Simulation)</div>', unsafe_allow_html=True)

        html_path = os.path.join(ENGINE_DIR, "dashboard.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_template = f.read()

            # stn_names: engine_wrapper에서 반환된 값 사용
            stn_names = cur.get("stn_names", sorted(list(
                st.session_state.bridge.stations.keys()
            )) if "bridge" in st.session_state else [])

            # dashboard.html에 필요한 데이터 구성 (기존 app.py 방식 그대로)
            dash_data = dict(cur)
            dash_data.update({
                "baseline":  BASELINE_MAP[st.session_state.ds_id],
                "stn_names": stn_names,
                "ds_name":   DS_LABELS[st.session_state.ds_id],
                "metadata":  st.session_state.data["metadata"] if "data" in st.session_state else {},
                "benchmark": st.session_state.benchmark_data,
                "breakdown": [
                    {"area":"Def_Met",    "mttf":10080,"mttr":35.28},
                    {"area":"Dielectric", "mttf":10080,"mttr":604.8},
                    {"area":"Diffusion",  "mttf":10080,"mttr":151.2},
                    {"area":"Dry_Etch",   "mttf":10080,"mttr":231.84},
                    {"area":"Implant",    "mttf":10080,"mttr":604.8},
                    {"area":"Litho",      "mttf":10080,"mttr":705.59},
                    {"area":"Litho_Met",  "mttf":10080,"mttr":35.28},
                    {"area":"Planar",     "mttf":10080,"mttr":201.6},
                    {"area":"TF",         "mttf":10080,"mttr":453.6},
                    {"area":"TF_Met",     "mttf":10080,"mttr":35.28},
                    {"area":"Wet_Etch",   "mttf":10080,"mttr":221.76},
                ],
                "products": [
                    {"name":"Product_1","priority":20,"wafers":25,"steps":522,"type":"HotLot"},
                    {"name":"Product_2","priority":20,"wafers":25,"steps":530,"type":"HotLot"},
                    {"name":"Product_3","priority":20,"wafers":25,"steps":584,"type":"HotLot"},
                    {"name":"Product_4","priority":20,"wafers":25,"steps":344,"type":"HotLot"},
                    {"name":"Product_5","priority":20,"wafers":25,"steps":530,"type":"HotLot"},
                    {"name":"Product_6","priority":10,"wafers":25,"steps":530,"type":"Regular"},
                    {"name":"Product_7","priority":10,"wafers":25,"steps":530,"type":"Regular"},
                    {"name":"Product_8","priority":10,"wafers":25,"steps":530,"type":"Regular"},
                    {"name":"Product_9","priority":10,"wafers":25,"steps":530,"type":"Regular"},
                    {"name":"Product_10","priority":10,"wafers":25,"steps":530,"type":"Regular"},
                    {"name":"Product_E1","priority":10,"wafers":1,"steps":522,"type":"Engineering"},
                    {"name":"Product_E2","priority":10,"wafers":2,"steps":530,"type":"Engineering"},
                    {"name":"Product_E3","priority":10,"wafers":2,"steps":584,"type":"Engineering"},
                ],
                "areas": [
                    {"name":"Diffusion","toolgroups":10},{"name":"Dry_Etch","toolgroups":21},
                    {"name":"Litho","toolgroups":11},{"name":"Implant","toolgroups":9},
                    {"name":"Dielectric","toolgroups":10},{"name":"Planar","toolgroups":6},
                    {"name":"TF","toolgroups":8},{"name":"Wet_Etch","toolgroups":7},
                    {"name":"Def_Met","toolgroups":7},{"name":"Litho_Met","toolgroups":4},
                    {"name":"TF_Met","toolgroups":3},
                ],
                "wip_history": cur.get("wip_history", []),
                "kpi_history": cur.get("kpi_history", []),
            })

            data_injection = f"const realData = {json.dumps(dash_data, ensure_ascii=False, default=str)};"
            final_html = html_template.replace("// [DATA_INJECTION_POINT]", data_injection)
            components.html(final_html, height=1500, scrolling=True)
        else:
            st.error(f"❌ dashboard.html 없음: {html_path}")
            st.info("A_APOS_Engine/dashboard.html 파일이 필요합니다. 기존 레포에서 복사해주세요.")

            # dashboard.html 없을 때 대체 시각화
            st.markdown('<div class="sec-header">대체 시각화 (dashboard.html 없을 때)</div>', unsafe_allow_html=True)
            wip_hist = cur.get("wip_history", [])
            if wip_hist:
                try:
                    import plotly.graph_objects as go
                    df_wip = pd.DataFrame(wip_hist)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_wip["tick"], y=df_wip["wip"],
                                              mode="lines", line=dict(color="#2563eb", width=2),
                                              fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
                    if "limit" in df_wip.columns:
                        fig.add_trace(go.Scatter(x=df_wip["tick"], y=df_wip["limit"],
                                                  mode="lines", name="WIP Limit",
                                                  line=dict(color="#ef4444", dash="dash")))
                    fig.update_layout(title="WIP 추이", height=250, margin=dict(l=40,r=20,t=40,b=40),
                                      plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("plotly 설치 필요: pip install plotly")

    # ════════════════════════════════════════════════════════
    # TAB 3: BENCHMARK ARENA
    # ════════════════════════════════════════════════════════
    with tab_bench:
        st.markdown('<div class="sec-header">🏆 Benchmark Arena — Dispatching Policy Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-blue">💡 사이드바에서 Policy 변경 → 🚀 정책 적용 → 시뮬레이션 진행 시 결과가 누적됩니다.</div>', unsafe_allow_html=True)

        # DS4 확정 실험 결과
        ref = {
            "FIFO":    {"ct":1071,"ontime":43.15,"cqt":1451,"hotlot":25.86},
            "EDD":     {"ct":909, "ontime":54.76,"cqt":1390,"hotlot":68.67},
            "Oracle":  {"ct":921, "ontime":66.74,"cqt":1255,"hotlot":64.91},
            "GNN+XGB": {"ct":748, "ontime":81.16,"cqt":880, "hotlot":91.30},
        }
        rows = []
        for pol, r in ref.items():
            rows.append({
                "Policy": f"★ {pol}" if pol=="GNN+XGB" else pol,
                "OTD (납기준수율)": f"{r['ontime']}%",
                "Avg CT": f"{r['ct']}h",
                "CQT 위반": f"{r['cqt']:,}건",
                "HotLot OTD": f"{r['hotlot']}%",
                "vs FIFO (CT)": f"{(r['ct']-1071)/1071*100:+.1f}%",
                "vs FIFO (OTD)": f"{r['ontime']-43.15:+.1f}%p",
            })
        df_ref = pd.DataFrame(rows)
        st.dataframe(df_ref, use_container_width=True, hide_index=True)

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("OTD 개선", "+38.0%p", "GNN+XGB vs FIFO")
        m2.metric("CT 단축", "-30.2%", "748h vs 1071h")
        m3.metric("CQT 위반 감소", "-39.4%", "880 vs 1451건")
        m4.metric("HotLot OTD", "+65.4%p", "91.3% vs 25.9%")

        if st.session_state.benchmark_data:
            st.markdown("---")
            st.markdown("**현재 세션 Live Benchmark**")
            live_rows = [{"Policy":p,"Avg CT (h)":f"{d.get('ct',0):.1f}","OTD (%)":f"{d.get('ontime',0):.1f}","WIP":f"{d.get('wip',0):,}"}
                         for p,d in st.session_state.benchmark_data.items()]
            st.dataframe(pd.DataFrame(live_rows), use_container_width=True, hide_index=True)

        # 고장 리스크 표
        st.markdown('<div class="sec-header">⚙️ 설비 고장 리스크 (MTTF / MTTR)</div>', unsafe_allow_html=True)
        bd = [
            {"Area":"Litho","MTTF":10080,"MTTR":705.59,"Availability":"93.5%","Risk":"🔴 HIGH"},
            {"Area":"Implant","MTTF":10080,"MTTR":604.8,"Availability":"94.3%","Risk":"🔴 HIGH"},
            {"Area":"Dielectric","MTTF":10080,"MTTR":604.8,"Availability":"94.3%","Risk":"🔴 HIGH"},
            {"Area":"TF","MTTF":10080,"MTTR":453.6,"Availability":"95.7%","Risk":"🟡 MED"},
            {"Area":"Planar","MTTF":10080,"MTTR":201.6,"Availability":"98.1%","Risk":"🟡 MED"},
            {"Area":"Dry_Etch","MTTF":10080,"MTTR":231.84,"Availability":"97.8%","Risk":"🟡 MED"},
            {"Area":"Wet_Etch","MTTF":10080,"MTTR":221.76,"Availability":"97.9%","Risk":"🟡 MED"},
            {"Area":"Diffusion","MTTF":10080,"MTTR":151.2,"Availability":"98.5%","Risk":"🟢 LOW"},
            {"Area":"Def_Met","MTTF":10080,"MTTR":35.28,"Availability":"99.7%","Risk":"🟢 LOW"},
            {"Area":"Litho_Met","MTTF":10080,"MTTR":35.28,"Availability":"99.7%","Risk":"🟢 LOW"},
            {"Area":"TF_Met","MTTF":10080,"MTTR":35.28,"Availability":"99.7%","Risk":"🟢 LOW"},
        ]
        st.dataframe(pd.DataFrame(bd), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # TAB 4: 재난 시뮬레이터
    # ════════════════════════════════════════════════════════
    with tab_disaster:
        st.markdown('<div class="sec-header">🚨 재난 시뮬레이터 — Resilience Score 산출</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-amber">⚡ <b>Resilience Score 산출 도구</b> — 교란 시나리오 발생 시 KPI 회복 속도를 정량화합니다. 발주처 실사 시 납기 신뢰도 증명 도구.</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1.2])
        with col_left:
            scenario_name = st.selectbox("재난 시나리오", list(DISASTER_SCENARIOS.keys()))
            scenario = DISASTER_SCENARIOS[scenario_name]
            extra_hotlot   = st.slider("긴급 HotLot 추가 투입", 0, 100, 0, 10)
            wip_reduction  = st.slider("WIP_CAP 즉시 감소율 (%)", 0, 50, 20, 5)
            duration_mult  = st.slider("장애 지속 시간 배율", 0.5, 3.0, 1.0, 0.5)

            if st.button("🚨 시나리오 실행", type="primary", use_container_width=True):
                actual_dur = int(scenario["duration"] * duration_mult)
                base_ct = BASELINE_MAP[st.session_state.ds_id]
                mttr_map = {"Litho":705.59,"Diffusion":151.2,"Implant":604.8}
                impact = min(actual_dur / (base_ct * 60) * 2.5, 0.8)
                resilience = max(0, min(100, round(100 - impact*60 - extra_hotlot*0.3 + wip_reduction*0.5)))
                st.session_state.disaster_log.append({
                    "scenario": scenario_name, "duration": actual_dur,
                    "extra_hotlot": extra_hotlot, "wip_reduction": wip_reduction,
                    "degraded_ct": round(base_ct*(1+impact*0.6)),
                    "degraded_otd": max(0, round(43.15-impact*40)),
                    "degraded_cqt": round(1451*(1+impact)),
                    "recovery_time": round(actual_dur*0.3 + mttr_map.get(scenario["area"],200)*2),
                    "resilience": resilience,
                })
                st.rerun()

        with col_right:
            if st.session_state.disaster_log:
                last = st.session_state.disaster_log[-1]
                res_color = "#22c55e" if last["resilience"]>=70 else "#f59e0b" if last["resilience"]>=40 else "#ef4444"
                st.markdown(f"""
                <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px;">
                    <div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:12px;">📋 {last['scenario']}</div>
                    <div style="text-align:center;margin-bottom:16px;">
                        <div style="font-size:52px;font-weight:700;color:{res_color};">{last['resilience']}</div>
                        <div style="font-size:13px;color:#64748b;">Resilience Score</div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                        <div style="background:#fef2f2;border-radius:8px;padding:10px;text-align:center;">
                            <div style="font-size:10px;color:#ef4444;font-weight:600;">CT 영향</div>
                            <div style="font-size:18px;font-weight:700;color:#ef4444;">{last['degraded_ct']}h</div>
                        </div>
                        <div style="background:#fef2f2;border-radius:8px;padding:10px;text-align:center;">
                            <div style="font-size:10px;color:#ef4444;font-weight:600;">OTD 영향</div>
                            <div style="font-size:18px;font-weight:700;color:#ef4444;">{last['degraded_otd']}%</div>
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
                cls = "alert-red" if last["resilience"]<50 else "alert-green"
                msg = "⚠️ Floodgate 자동 발동 권장 — WIP_CAP 즉시 감소" if last["resilience"]<50 else "✅ 공장 회복력 양호 — 납기 신뢰도 유지 가능"
                st.markdown(f'<div class="{cls}" style="margin-top:10px;">{msg}</div>', unsafe_allow_html=True)

        if len(st.session_state.disaster_log) > 1:
            st.markdown("---")
            st.markdown("**시나리오 비교 이력**")
            df_dis = pd.DataFrame(st.session_state.disaster_log)[
                ["scenario","duration","degraded_ct","degraded_otd","degraded_cqt","resilience"]]
            df_dis.columns = ["시나리오","장애시간(min)","CT(h)","OTD(%)","CQT위반","Resilience"]
            st.dataframe(df_dis, use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-header">🔩 Floodgate Control — TOC Buffer Management</div>', unsafe_allow_html=True)
        st.markdown("""
        | 버퍼 상태 | 조건 | Floodgate 조치 | Little's Law 근거 |
        |---|---|---|---|
        | 🟢 Green | Down = 0 | WIP_CAP 원래 값 | 정상 λ 유지 |
        | 🟡 Yellow | Down = 1 | WIP_CAP 5% 감소 | μ 5% 감소 → λ 조절 |
        | 🔴 Red | Down ≥ 2 | WIP_CAP 20% 감소 | μ 20% 감소 → W 발산 방지 |
        """)

    # ════════════════════════════════════════════════════════
    # TAB 5: AI EXPLAINABILITY
    # ════════════════════════════════════════════════════════
    with tab_xai:
        st.markdown('<div class="sec-header">🔍 AI Explainability — GNN + XGBoost</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-blue">🧠 RL 블랙박스와 달리 모든 의사결정에 설명이 가능합니다. GNN Attention + XGBoost SHAP으로 "왜 이 Lot을 먼저?"를 추적합니다.</div>', unsafe_allow_html=True)

        col_g, col_x = st.columns(2)
        with col_g:
            st.markdown("**DS별 GNN 피처 전략**")
            st.dataframe(pd.DataFrame({
                "DS": ["DS1","DS2","DS3","DS4"],
                "지배적 피처": ["CR, waiting_time, priority","CR, remaining_steps, station_queue",
                               "area_down_rate, cqt_urgency, cascade_risk","All 13 features"],
                "전략": ["기본 4개","중간 7개","고장 피처 추가","전체 통합"],
            }), use_container_width=True, hide_index=True)
            st.markdown("**CQT Urgency (EWS 개념)**")
            st.code("""CQT_urgency = CQT잔여시간 / 설비평균처리시간
urgency < 2.0 → CRITICAL  (+100점)
urgency < 5.0 → URGENT    (+50점)
→ Priority 역전 발생 (동적 위기 관리)""")

        with col_x:
            st.markdown("**XGBoost SHAP Top-3 (DS별 예상)**")
            st.dataframe(pd.DataFrame({
                "DS": ["DS1","DS2","DS3","DS4"],
                "SHAP #1": ["critical_ratio","station_queue","area_down_rate","cascade_risk"],
                "SHAP #2": ["waiting_time","remaining_steps","cqt_urgency","WIP_slope"],
                "SHAP #3": ["priority","cqt_urgency","cascade_risk","cqt_urgency"],
            }), use_container_width=True, hide_index=True)
            st.markdown("**Sigmoid 변환 — 98% 포화 해결**")
            st.code("""# 기존: WIP(150)*0.5 + util(0.8)*0.3 = 75.24 → 98% 포화
# 수정:
norm_wip = min(wip / 100.0, 1.0)      # [0,1] 정규화
raw = norm_wip*0.4 + util*0.3 + slope*0.2
# Sigmoid: P = 1/(1+e^(-5*(raw-0.5)))
# 결과: 5%~95% 범위에서 변별력 있게 분포""")

        gnn_logs = cur.get("gnn_logs", [])
        if gnn_logs:
            st.markdown('<div class="sec-header">🤖 GNN + XGBoost Action Log</div>', unsafe_allow_html=True)
            log_html = '<div style="background:#0f172a;border-radius:10px;padding:14px;font-family:monospace;font-size:11px;max-height:200px;overflow-y:auto;">'
            for log in gnn_logs[-20:]:
                color = "#ef4444" if "CRITICAL" in log or "Warning" in log else "#22c55e" if "Resolved" in log else "#60a5fa" if "GNN" in log else "#fbbf24"
                log_html += f'<div style="color:{color};margin-bottom:3px;">▸ {log}</div>'
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# LLM PANEL
# ════════════════════════════════════════════════════════
with llm_col:
    st.markdown("""
    <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;">
        <div style="font-size:13px;font-weight:600;color:#0f172a;border-bottom:1px solid #e2e8f0;padding-bottom:10px;margin-bottom:12px;">
            🤖 A-APOS Assistant
        </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", label_visibility="collapsed")

    if not st.session_state.chat_history:
        st.markdown("""<div class="chat-ai">안녕하세요! 빠른 질문을 눌러보세요.<br><br>• Litho 병목 원인<br>• GNN vs FIFO<br>• 재난 발생 대응<br>• CQT urgency 설명</div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history[-6:]:
            css = "chat-user" if msg["role"]=="user" else "chat-ai"
            st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("**빠른 질문**")
    fallback_map = {
        "Litho 병목 원인": "Litho 구역은 MTTR 705.59분으로 전체 구역 중 최고 위험입니다. 고장 시 Cascade_risk(Litho_down × Implant_WIP_ROC)가 급증하며 Implant 구역으로 연쇄 전파됩니다. Floodgate 발동(WIP_CAP 20% 감소) 권장.",
        "GNN vs FIFO 차이": "FIFO는 현재 대기열만 보지만, GNN은 구역→구역 엣지로 미래 경합 경로를 예측합니다. CQT urgency < 2.0 시 +100점으로 우선순위 즉시 역전. DS4 결과: OTD +38%p, CT -30% 달성.",
        "재난 발생 대응": "1) Floodgate 자동 발동 (Down≥2 → WIP_CAP 20% 감소) 2) XGBoost 경고 확인 (70% 임계값) 3) 재난 시뮬레이터 탭에서 Resilience Score 산출 4) What-if 파라미터 조정.",
        "CQT urgency 설명": "CQT_urgency = CQT잔여시간 / 설비평균처리시간. 의료 ICU EWS 개념 이식. urgency < 2.0 → CRITICAL (+100점), < 5.0 → URGENT (+50점). 정적 우선순위에서 동적 위기관리로 전환.",
    }
    for q in fallback_map:
        if st.button(q, key=f"qq_{q}", use_container_width=True):
            st.session_state.chat_history.append({"role":"user","content":q})
            if api_key:
                try:
                    import openai
                    client = openai.OpenAI(api_key=api_key)
                    ctx = f"WIP={cur.get('wip',0)}, OTD={cur.get('kpi',{}).get('ontime_pct',0)}%, CT={cur.get('kpi',{}).get('avg_ct',0)}h, CQT위반={cur.get('cqt',{}).get('violations',0)}건, Dataset={DS_LABELS[st.session_state.ds_id]}"
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":f"SMT FAB A-APOS 전문 어시스턴트. 현재 상태:{ctx}. 200자 이내 실무적 답변."}] +
                                  st.session_state.chat_history[-4:],
                        max_tokens=300, temperature=0.3,
                    )
                    answer = resp.choices[0].message.content
                except Exception as e:
                    answer = f"API 오류: {str(e)[:60]}"
            else:
                answer = fallback_map[q]
            st.session_state.chat_history.append({"role":"assistant","content":answer})
            st.rerun()

    user_input = st.text_input("직접 질문", placeholder="공장 상태에 대해 질문...", label_visibility="collapsed", key="chat_input")
    if st.button("전송", use_container_width=True) and user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        if api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"SMT FAB A-APOS 전문 어시스턴트. 200자 이내."}] +
                              st.session_state.chat_history[-4:],
                    max_tokens=300, temperature=0.3,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"오류: {str(e)[:60]}"
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