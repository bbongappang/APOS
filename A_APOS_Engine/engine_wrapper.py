"""
engine_wrapper.py — SimPy ↔ Streamlit 브리지
Floodgate Control (조건부 WIP 상한) 포함
"""
import simpy
import random
import numpy as np
import pandas as pd
from datetime import datetime
from .factory_engine import AdvancedStation, Lot, failure_process
from .gnn_dispatcher import GNNDispatcher
from .xgb_predictor import XGBBottleneckPredictor

BASE_DATE = datetime(2018, 1, 1)

BREAKDOWN_TABLE = {
    "Def_Met":    (10080, 35.28),
    "Dielectric": (10080, 604.8),
    "Diffusion":  (10080, 151.2),
    "Dry_Etch":   (10080, 231.84),
    "Implant":    (10080, 604.8),
    "Litho":      (10080, 705.59),
    "Litho_Met":  (10080, 35.28),
    "Planar":     (10080, 201.6),
    "TF":         (10080, 453.6),
    "TF_Met":     (10080, 35.28),
    "Wet_Etch":   (10080, 221.76),
}

PTPER_TO_UNIT = {
    "per_batch": "Batch", "per_piece": "Wafer", "per_lot": "Lot",
    "Batch": "Batch", "Wafer": "Wafer", "Lot": "Lot",
}


def _sf(val, default=0.0) -> float:
    try:
        s = str(val).strip()
        return default if s in ('', 'nan', 'NaN', 'None') else float(s)
    except (TypeError, ValueError):
        return default


def _ss(val) -> str:
    s = str(val).strip() if val is not None else ''
    return '' if s in ('nan', 'NaN', 'None') else s


def _get_area(stn_name: str, area_hint: str = "") -> str:
    h = area_hint.strip()
    if h and h in BREAKDOWN_TABLE:
        return h
    n = stn_name.lower()
    for key, area in [
        ("diffusion", "Diffusion"), ("de_", "Dry_Etch"), ("dry_etch", "Dry_Etch"),
        ("litho_met", "Litho_Met"), ("lithomet", "Litho_Met"),
        ("litho_be", "Litho"), ("litho", "Litho"),
        ("implant", "Implant"), ("epi", "Implant"),
        ("dielectric", "Dielectric"), ("planar", "Planar"), ("cmp", "Planar"),
        ("tf_met", "TF_Met"), ("tf_", "TF"), ("tf", "TF"),
        ("we_", "Wet_Etch"), ("wet_etch", "Wet_Etch"),
        ("defmet", "Def_Met"), ("def_met", "Def_Met"),
    ]:
        if key in n:
            return area
    return "Dry_Etch"


def _parse_route_df(df: pd.DataFrame) -> list:
    steps = []
    col = {c.upper().strip(): c for c in df.columns}
    stn_col   = col.get('STNFAM')
    ptime_col = col.get('PTIME')
    ptper_col = col.get('PTPER')
    bmin_col  = col.get('BATCHMN')
    bmax_col  = col.get('BATCHMX')
    setup_col = col.get('SETUP')
    stime_col = col.get('STIME')
    ignore_col= col.get('IGNORE')

    if not stn_col or not ptime_col:
        return steps

    for _, row in df.iterrows():
        stn = _ss(row.get(stn_col, ''))
        if not stn:
            continue
        ptime     = max(0.01, _sf(row.get(ptime_col), 1.0))
        ptper_raw = _ss(row.get(ptper_col, 'per_lot')) if ptper_col else 'per_lot'
        proc_unit = PTPER_TO_UNIT.get(ptper_raw, 'Lot')
        setup_req = _ss(row.get(setup_col, '')) if setup_col else ''
        setup_cost= _sf(row.get(stime_col), 0.0) if stime_col else 0.0
        bmin      = _sf(row.get(bmin_col), 0.0) if bmin_col else 0.0
        bmax      = _sf(row.get(bmax_col), bmin) if bmax_col else bmin
        area_hint = _ss(row.get(ignore_col, '')) if ignore_col else ''
        steps.append({
            "station": stn, "ptime": ptime, "proc_unit": proc_unit,
            "setup": setup_req, "setup_cost": setup_cost,
            "is_batch": proc_unit == "Batch",
            "batch_min_wafers": int(bmin) if bmin > 0 else 1,
            "batch_max_wafers": int(bmax) if bmax > 0 else 1,
            "area": _get_area(stn, area_hint),
        })
    return steps


class SimBridge:
    def __init__(self, env: simpy.Environment, data: dict, overrides: dict = None):
        self.env            = env
        self.data           = data
        self.overrides      = overrides or {}
        self.stations:      dict[str, AdvancedStation] = {}
        self.active_lots:   list = []
        self.completed_lots:list = []

        policy = self.overrides.get("policy", "GNN")
        self.gnn_dispatcher = GNNDispatcher(policy=policy)
        self.xgb_predictor  = XGBBottleneckPredictor(threshold=0.7)

        self.kpi_tracker = {"completed": 0, "cycle_times": [], "ontime_count": 0}
        self.wip_history: list = []
        self.kpi_history: list = []
        self._stn_area:   dict[str, str] = {}
        self.release_events: list = []
        self.gate_control_active = False
        self.gate_logs = []

        # Route 파싱
        self.route_steps: dict[str, list] = {}
        for key, df in data["routes"].items():
            steps = _parse_route_df(df)
            if steps:
                self.route_steps[key] = steps
                for s in steps:
                    self._stn_area[s["station"]] = s["area"]

        # 설비 초기화
        stn_cfg: dict[str, dict] = {}
        for steps in self.route_steps.values():
            for s in steps:
                name = s["station"]
                if name not in stn_cfg:
                    stn_cfg[name] = {"is_batch": False, "batch_min_wafers": 1, "batch_max_wafers": 1}
                if s["is_batch"]:
                    stn_cfg[name]["is_batch"] = True
                    cur = stn_cfg[name]["batch_min_wafers"]
                    if cur == 1 or s["batch_min_wafers"] < cur:
                        stn_cfg[name]["batch_min_wafers"] = s["batch_min_wafers"]
                        stn_cfg[name]["batch_max_wafers"] = s["batch_max_wafers"]

        tool_capacity = data.get("tool_capacity", {})
        cap_factor = self.overrides.get("capacity_factor", 1.0)

        stn_ptimes = {}
        for steps in self.route_steps.values():
            for s in steps:
                stn_ptimes.setdefault(s["station"], []).append(s["ptime"])

        for name, cfg in stn_cfg.items():
            raw_cap  = tool_capacity.get(name, 1)
            capacity = max(1, int(raw_cap * cap_factor))
            self.stations[name] = AdvancedStation(
                env, name, capacity=capacity,
                is_batch=cfg["is_batch"],
                batch_min_wafers=cfg["batch_min_wafers"],
                batch_max_wafers=cfg["batch_max_wafers"],
                dispatcher=self.gnn_dispatcher,
            )
            if name in stn_ptimes:
                self.stations[name].mean_ptime = np.mean(stn_ptimes[name])

        # 고장 프로세스 (DS3, DS4)
        if data.get("downs") is not None:
            mttf_factor = self.overrides.get("mttf_factor", 1.0)
            mttr_factor = self.overrides.get("mttr_factor", 1.0)
            area_bd = self._parse_downcal(data["downs"])
            for stn_name, stn_obj in self.stations.items():
                area = self._stn_area.get(stn_name, "Dry_Etch")
                mttf, mttr = area_bd.get(area, BREAKDOWN_TABLE.get(area, (10080, 200)))
                env.process(failure_process(env, stn_obj, mttf * mttf_factor, mttr * mttr_factor))

        if self.route_steps:
            env.process(self._release_controller())
            env.process(self._cqt_monitor())

    def _cqt_monitor(self):
        while True:
            yield self.env.timeout(1.0)

    def _parse_downcal(self, downs_df) -> dict:
        result = {}
        if downs_df is None or downs_df.empty:
            return result
        col = {c.upper().strip(): c for c in downs_df.columns}
        ignore_col = col.get('IGNORE')
        mttf_col   = col.get('MTTF')
        mttr_col   = col.get('MTTR')
        if not (ignore_col and mttf_col and mttr_col):
            return result
        for _, row in downs_df.iterrows():
            area = _ss(row.get(ignore_col, ''))
            mttf = _sf(row.get(mttf_col), 10080)
            mttr = _sf(row.get(mttr_col), 200)
            if area:
                result[area] = (mttf, mttr)
        return result

    def _lot_process(self, lot: Lot, steps: list):
        lot.total_steps  = len(steps)
        lot.current_step = 0
        for step in steps:
            stn_name = step["station"]
            if stn_name not in self.stations:
                lot.current_step += 1
                continue
            lot.current_station = stn_name
            lot.current_step   += 1
            yield self.env.process(
                self.stations[stn_name].process(
                    lot, step["ptime"], step["setup"],
                    step["setup_cost"], step["proc_unit"],
                )
            )
        lot.finish_time = self.env.now
        lot.current_station = "DONE"
        ct = lot.finish_time - lot.start_time
        ok = (lot.due_date is None) or (lot.finish_time <= lot.due_date)
        self.kpi_tracker["completed"]    += 1
        self.kpi_tracker["cycle_times"].append(ct)
        self.kpi_tracker["ontime_count"] += int(ok)
        if lot in self.active_lots:
            self.active_lots.remove(lot)
        self.completed_lots.append(lot)

    def _release_controller(self):
        BASE_WIP_LIMIT = self.overrides.get("wip_limit", 3000)
        orders = self.data.get("orders", pd.DataFrame())
        rkeys  = list(self.route_steps.keys())
        if orders.empty or not rkeys:
            yield self.env.timeout(0)
            return
        col = {c.upper().strip(): c for c in orders.columns}
        for _, row in orders.iterrows():
            lot_name  = _ss(row.get(col.get('LOT', ''), 'LOT'))
            part      = _ss(row.get(col.get('PART', ''), 'part_1'))
            priority  = int(_sf(row.get(col.get('PRIOR', ''), 10), 10))
            wafers    = int(_sf(row.get(col.get('PIECES', ''), 25), 25))
            start_min = _sf(row.get(col.get('START_MIN', ''), 0), 0.0)
            due_min   = _sf(row.get(col.get('DUE_MIN', ''), 99999), 99999.0)
            repeat    = _sf(row.get(col.get('REPEAT', ''), 258.46), 258.46)
            route_key = self._find_route(part, rkeys)
            self.env.process(
                self._repeat_release(lot_name, part, priority, wafers,
                                     start_min, due_min, repeat, route_key, BASE_WIP_LIMIT)
            )
        yield self.env.timeout(0)

    def _get_dynamic_wip_limit(self, base_limit: int) -> int:
        """Floodgate Control — TOC Buffer Management 기반"""
        area_down_counts = {}
        for name, stn in self.stations.items():
            if stn.is_down:
                area = self._stn_area.get(name, "Unknown")
                area_down_counts[area] = area_down_counts.get(area, 0) + 1

        critical_areas = ["Litho", "Implant", "Diffusion"]
        max_down_impact = 0
        culprit_area = ""
        for area in critical_areas:
            down_count = area_down_counts.get(area, 0)
            if down_count >= 2:      # RED: 20% 감소
                max_down_impact = max(max_down_impact, 0.2)
                culprit_area = area
            elif down_count >= 1:    # YELLOW: 5% 감소
                max_down_impact = max(max_down_impact, 0.05)
                if not culprit_area:
                    culprit_area = area

        if max_down_impact > 0:
            current_limit = int(base_limit * (1.0 - max_down_impact))
            if not self.gate_control_active:
                self.gate_control_active = True
                self.gate_logs.append(
                    f"🚧 [Floodgate] ACTIVE: {culprit_area} Down → WIP_CAP {base_limit} → {current_limit} ({max_down_impact*100:.0f}% 감소)"
                )
        else:
            current_limit = base_limit
            if self.gate_control_active:
                self.gate_control_active = False
                self.gate_logs.append(
                    f"🔓 [Floodgate] DEACTIVATED: 전 구역 복구 → WIP_CAP 원래 값 {base_limit} 복원"
                )
        return current_limit

    def _find_route(self, part: str, rkeys: list) -> str:
        num = part.split('_')[-1]
        target = f"part_{num}"
        if target in self.route_steps:
            return target
        for rk in rkeys:
            if rk.endswith(f"_{num}"):
                return rk
        if part in self.route_steps:
            return part
        return rkeys[0]

    def _repeat_release(self, lot_name, part, priority, wafers,
                        start_min, due_min, repeat_interval, route_key, wip_limit):
        delay = max(0.0, start_min - self.env.now)
        if delay > 0:
            yield self.env.timeout(delay)
        counter = 0
        lot_due_duration = max(1.0, due_min - start_min)
        while True:
            current_limit = self._get_dynamic_wip_limit(wip_limit)
            while len(self.active_lots) >= current_limit:
                yield self.env.timeout(repeat_interval)
                current_limit = self._get_dynamic_wip_limit(wip_limit)
            lot = Lot(
                lot_id=f"{lot_name}_{counter:05d}", part=part,
                start_time=self.env.now, priority=priority,
                wafers=wafers, due_date=self.env.now + lot_due_duration,
            )
            lot.cqt_deadline = self.env.now + (lot_due_duration * 0.8)
            counter += 1
            self.active_lots.append(lot)
            self.release_events.append(self.env.now)
            if len(self.release_events) > 1000:
                self.release_events = [t for t in self.release_events if t > self.env.now - 60]
            self.env.process(self._lot_process(lot, self.route_steps[route_key]))
            yield self.env.timeout(repeat_interval)

    def update_ui_state(self) -> dict:
        stn_states = []
        area_stats: dict[str, dict] = {}
        gnn_logs = []

        for name, stn in self.stations.items():
            state = stn.state
            area  = self._stn_area.get(name, "Dry_Etch")
            stn_states.append({"id": name, "state": state, "util": stn.utilization, "area": area})
            if hasattr(stn, 'action_logs') and stn.action_logs:
                gnn_logs.extend(stn.action_logs)
                stn.action_logs = []
            if area not in area_stats:
                area_stats[area] = {"busy":0,"down":0,"setup":0,"idle":0,"total":0,"wip":0}
            area_stats[area][state] += 1
            area_stats[area]["total"] += 1

        area_utils_sum: dict[str, list] = {}
        for name, stn in self.stations.items():
            area = self._stn_area.get(name, "Dry_Etch")
            area_utils_sum.setdefault(area, []).append(stn.utilization)

        cqt_violations = 0
        urgent_lots = []
        for lot in self.active_lots:
            area = self._stn_area.get(lot.current_station, "Unknown")
            if area in area_stats:
                area_stats[area]["wip"] += 1
            if lot.cqt_deadline:
                if self.env.now > lot.cqt_deadline:
                    cqt_violations += 1
                elif lot.cqt_deadline - self.env.now < 120:
                    urgent_lots.append({
                        "id": lot.id, "area": area,
                        "rem": round(lot.cqt_deadline - self.env.now, 1)
                    })

        for area, utils in area_utils_sum.items():
            if area in area_stats:
                area_stats[area]["avg_util"] = round(float(np.mean(utils)), 1)

        completed  = self.kpi_tracker["completed"]
        cts        = self.kpi_tracker["cycle_times"]
        avg_ct     = round(float(np.mean(cts)) / 60.0, 1) if cts else 0.0
        ontime_pct = round(self.kpi_tracker["ontime_count"] / completed * 100, 1) if completed > 0 else 0.0
        down_count = sum(1 for s in stn_states if s["state"] == "down")
        wip        = len(self.active_lots)
        tick       = int(self.env.now)
        current_wip_limit = self._get_dynamic_wip_limit(self.overrides.get("wip_limit", 3000))

        self.wip_history.append({"tick": tick, "wip": wip, "limit": current_wip_limit})
        self.kpi_history.append({"tick": tick, "ct": avg_ct, "ontime": ontime_pct})
        if len(self.wip_history) > 60: self.wip_history = self.wip_history[-60:]
        if len(self.kpi_history) > 60: self.kpi_history = self.kpi_history[-60:]

        snapshot = {"area_stats": area_stats, "wip": wip, "tick": self.env.now}
        xgb_probs, xgb_logs = self.xgb_predictor.predict(snapshot)

        gnn_logs.extend(xgb_logs)
        gnn_logs.extend(self.gate_logs)
        self.gate_logs = []

        # stn_names: 모든 설비 이름 목록
        stn_names = sorted(list(self.stations.keys()))

        return {
            "tick": tick, "wip": wip,
            "stations": stn_states, "area_stats": area_stats,
            "stn_names": stn_names,
            "gnn_logs": gnn_logs,
            "xgb_probs": xgb_probs,
            "kpi": {"completed": completed, "avg_ct": avg_ct,
                    "ontime_pct": ontime_pct, "down_count": down_count},
            "cqt": {"violations": cqt_violations, "urgent_count": len(urgent_lots),
                    "urgent_list": urgent_lots[:10]},
            "wip_history": self.wip_history[-30:],
            "kpi_history": self.kpi_history[-30:],
        }

    def run_step(self, until: int) -> dict:
        self.env.run(until=until)
        return self.update_ui_state()

    def force_station_down(self, station_name: str, duration: float):
        if station_name in self.stations:
            self.env.process(self._manual_down(self.stations[station_name], duration))

    def _manual_down(self, stn: AdvancedStation, duration: float):
        stn.is_down = True
        stn.stats["down_time"] += duration
        yield self.env.timeout(duration)
        stn.is_down = False

    def get_summary(self) -> dict:
        total = len(self.stations)
        down  = sum(1 for s in self.stations.values() if s.state == "down")
        busy  = sum(1 for s in self.stations.values() if s.state == "busy")
        return {
            "total_stations": total, "busy": busy, "down": down,
            "idle": total - busy - down, "wip": len(self.active_lots),
            "completed": self.kpi_tracker["completed"],
        }
