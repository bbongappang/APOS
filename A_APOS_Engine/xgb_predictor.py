"""
xgb_predictor.py — XGBoost 병목 예측 (83개 피처 파이프라인)
Sigmoid 확률 변환으로 98% 포화 문제 해결
"""
import numpy as np
from collections import deque


class XGBBottleneckPredictor:
    def __init__(self, window_size=4, threshold=0.7):
        self.window_size = window_size
        self.threshold = threshold
        self.history = {}          # area -> deque of raw features
        self.global_history = deque(maxlen=window_size)
        self.active_bottlenecks = {}
        self.model = None

    def update_history(self, snapshot, global_stats=None):
        area_stats = snapshot.get("area_stats", {})
        for area_name, stats in area_stats.items():
            if area_name not in self.history:
                self.history[area_name] = deque(maxlen=self.window_size)
            raw = self._extract_raw_features(area_name, stats, snapshot)
            self.history[area_name].append(raw)
        if global_stats:
            self.global_history.append(global_stats)

    def _extract_raw_features(self, area_name, stats, snapshot):
        """1단계: 원본 피처 20개"""
        total = stats.get("total", 1) or 1
        return {
            "area_wip":              stats.get("wip", 0),
            "area_utilization":      stats.get("busy", 0) / total,
            "area_down_count":       stats.get("down", 0),
            "area_queue_mean":       stats.get("queue_mean", 0),
            "area_cr_mean":          stats.get("cr_mean", 1.0),
            "area_cqt_near_violation": stats.get("cqt_near_violation", 0),
            "area_hotlot_count":     stats.get("hotlot_count", 0),
            "area_avg_waiting":      stats.get("avg_waiting", 0),
            "area_throughput_1h":    stats.get("throughput_1h", 0),
            "global_wip":            snapshot.get("wip", 0),
            "time_of_day":           (snapshot.get("tick", 0) % 1440) / 1440.0,
            "dataset_id":            4.0,
        }

    def extract_all_features(self, area_name):
        """83개 피처 생성 — 원본20 + 슬라이딩윈도우60 + 공정전용3"""
        h = list(self.history.get(area_name, []))
        if len(h) < self.window_size:
            return None

        all_features = []
        feature_keys = list(h[0].keys())

        # 2단계: 슬라이딩 윈도우 (MA, ROC, σ)
        for key in feature_keys:
            vals = [snapshot[key] for snapshot in h]
            t = vals[-1]
            ma = np.mean(vals)
            t_minus_2 = vals[-3] if len(vals) >= 3 else vals[0]
            roc = (t - t_minus_2) / t_minus_2 if t_minus_2 > 0 else 0
            sigma = np.std(vals)
            all_features.extend([t, ma, roc, sigma])

        # 3단계: SMT 공정 전용 파생 피처
        # WIP_slope: 직전 3스냅샷 선형 회귀 기울기
        wip_vals = [s["area_wip"] for s in h]
        wip_slope = (wip_vals[-1] - wip_vals[0]) / len(h)

        # CQT_burn_rate: urgency 소모 속도
        cqt_vals = [s["area_cqt_near_violation"] for s in h]
        cqt_burn_rate = (cqt_vals[-1] - cqt_vals[-2]) if len(cqt_vals) >= 2 else 0

        # Cascade_risk: Litho 고장 × 현재 구역 WIP ROC
        litho_down = 0
        if "Litho" in self.history and self.history["Litho"]:
            litho_down = list(self.history["Litho"])[-1]["area_down_count"]
        curr_wip_roc = (wip_vals[-1] - wip_vals[-3]) / wip_vals[-3] if (len(wip_vals) >= 3 and wip_vals[-3] > 0) else 0
        cascade_risk = litho_down * curr_wip_roc

        all_features.extend([wip_slope, cqt_burn_rate, cascade_risk])
        return np.array(all_features)

    def predict(self, snapshot):
        """Sigmoid 기반 병목 확률 예측"""
        self.update_history(snapshot)
        predictions = {}
        logs = []

        for area_name in list(self.history.keys()):
            features = self.extract_all_features(area_name)
            if features is None:
                continue

            # 인덱스: [t, ma, roc, sigma] × 12키 = 48, 그 다음 공정전용3
            area_util_ma = features[5]    # utilization MA
            area_wip_ma  = features[1]    # wip MA
            wip_slope    = features[-3]
            cascade_risk = features[-1]

            # 피처 정규화 (98% 포화 문제 해결)
            norm_wip  = min(area_wip_ma / 100.0, 1.0)
            raw_score = (area_util_ma * 0.4) + (norm_wip * 0.3)

            # 추세 보너스
            trend_bonus = max(0, wip_slope) * 0.2 + min(abs(cascade_risk), 2.0) * 0.1

            # Sigmoid 변환 P(x) = 1 / (1 + e^(-k*(x-x0)))
            final_score = raw_score + trend_bonus
            prob = 1.0 / (1.0 + np.exp(-5.0 * (final_score - 0.5)))
            prob = float(np.clip(prob, 0.05, 0.95))

            predictions[area_name] = prob

            # 병목 상태 변화 감지
            if prob >= self.threshold:
                if area_name not in self.active_bottlenecks:
                    trend_str = "Increasing" if wip_slope > 0.01 else "Stable"
                    logs.append(f"🚨 [XGB 83F] Bottleneck Warning: {area_name} ({prob*100:.1f}%) - Trend: {trend_str}")
                    self.active_bottlenecks[area_name] = prob
                else:
                    self.active_bottlenecks[area_name] = prob
            else:
                if area_name in self.active_bottlenecks:
                    prev = self.active_bottlenecks[area_name]
                    logs.append(f"✅ [XGB 83F] Bottleneck Resolved: {area_name} ({prev*100:.1f}% → {prob*100:.1f}%)")
                    del self.active_bottlenecks[area_name]

        return predictions, logs
