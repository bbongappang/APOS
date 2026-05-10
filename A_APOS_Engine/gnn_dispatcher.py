"""
gnn_dispatcher.py — GNN 기반 실시간 디스패칭
CQT-aware CR Oracle + EWS(Early Warning Score) 통합
"""


class GNNDispatcher:
    def __init__(self, policy="GNN"):
        self.policy = policy
        self.model = None

    def compute_scores(self, station, waiting_lots):
        if not waiting_lots:
            return {}

        now = station.env.now
        scores = {}

        for lot in waiting_lots:
            if self.policy == "FIFO":
                scores[lot.id] = 1000000 - lot.start_time

            elif self.policy == "EDD":
                due = lot.due_date if lot.due_date else 999999
                scores[lot.id] = 1000000 - due

            elif self.policy == "CR":
                due = lot.due_date if lot.due_date else 999999
                rem_time = max(1, (lot.total_steps - lot.current_step) * station.mean_ptime)
                cr = (due - now) / rem_time
                scores[lot.id] = 1000 - cr

            else:  # GNN — CQT-aware CR Oracle
                # 기본 점수: CR 기반
                due = lot.due_date if lot.due_date else 999999
                rem_time = max(1, (lot.total_steps - lot.current_step) * station.mean_ptime)
                cr = (due - now) / rem_time
                base_score = max(0, 500 - cr * 10)

                # HotLot 가산
                hotlot_bonus = 50 if lot.priority >= 20 else 0

                # Setup 연속성 가산
                setup_bonus = 30 if lot.setup_req == station.current_setup else 0

                # CQT Urgency — EWS 개념 (핵심 novelty)
                urgency = lot.get_cqt_urgency(now, station.mean_ptime)
                if urgency < 2.0:
                    urgency_bonus = 100   # CRITICAL
                elif urgency < 5.0:
                    urgency_bonus = 50    # URGENT
                else:
                    urgency_bonus = 0

                scores[lot.id] = base_score + hotlot_bonus + setup_bonus + urgency_bonus

        return scores

    def select_best_lot(self, station, waiting_lots):
        if not waiting_lots:
            return None
        scores = self.compute_scores(station, waiting_lots)
        best_lot = max(waiting_lots, key=lambda l: scores.get(l.id, 0))
        return best_lot, scores
