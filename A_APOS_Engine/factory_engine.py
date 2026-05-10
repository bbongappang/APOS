"""
factory_engine.py — A-APOS SimPy 핵심 엔진
"""
import simpy
import random


class Lot:
    def __init__(self, lot_id, part, start_time, priority,
                 wafers=25, setup_req="", due_date=None):
        self.id = lot_id
        self.part = part
        self.start_time = start_time
        self.priority = priority
        self.wafers = wafers
        self.setup_req = setup_req
        self.due_date = due_date
        self.cqt_deadline = None
        self.current_step = 0
        self.total_steps = 0
        self.current_station = None
        self.finish_time = None
        self.wait_event = None
        self.batch_done_ev = None
        self.is_leader = False

    @property
    def is_tardy(self) -> bool:
        if self.due_date is None or self.finish_time is None:
            return False
        return self.finish_time > self.due_date

    def get_cqt_urgency(self, now: float, mean_ptime: float) -> float:
        if self.cqt_deadline is None or mean_ptime <= 0:
            return 999.0
        rem = max(0, self.cqt_deadline - now)
        return rem / mean_ptime


class AdvancedStation:
    BATCH_WAIT_MAX = 200.0

    def __init__(self, env, name, capacity=1,
                 is_batch=False, batch_min_wafers=1, batch_max_wafers=1,
                 dispatcher=None):
        self.env = env
        self.name = name
        self.res = simpy.PriorityResource(env, capacity=capacity)
        self.is_batch = is_batch
        self.batch_min_wafers = max(1, batch_min_wafers)
        self.batch_max_wafers = max(1, batch_max_wafers)
        self.batch_queue = []
        self.waiting_lots = []
        self.dispatcher = dispatcher
        self.batch_done_event = None
        self.is_down = False
        self.last_failure_time = 0.0
        self.last_recovery_time = 0.0
        self.current_setup = None
        self.mean_ptime = 1.0
        self.stats = {"util_time": 0.0, "setup_time": 0.0, "down_time": 0.0, "lots_processed": 0}
        self.action_logs = []

    @property
    def state(self) -> str:
        if self.is_down:
            return "down"
        if self.res.count > 0:
            return "busy"
        if self.is_batch and self.batch_queue:
            return "setup"
        return "idle"

    @property
    def queue_size(self) -> int:
        return len(self.batch_queue) + len(self.res.queue) + len(self.waiting_lots)

    @property
    def time_since_last_failure(self) -> float:
        return self.env.now - self.last_failure_time

    @property
    def utilization(self) -> float:
        now = self.env.now
        return round(self.stats["util_time"] / now * 100, 1) if now > 0 else 0.0

    @property
    def queued_wafers(self) -> int:
        return sum(lot.wafers for lot in self.batch_queue)

    def _release_batch(self):
        waiting = list(self.batch_queue)
        self.batch_queue = []
        for l in waiting:
            if l.wait_event and not l.wait_event.triggered:
                l.wait_event.succeed()

    def process(self, lot, ptime: float, setup_req: str,
                setup_cost: float, proc_unit: str = "Lot"):
        while self.is_down:
            yield self.env.timeout(30)

        if proc_unit == "Wafer":
            actual_ptime = max(0.01, ptime * lot.wafers)
        else:
            actual_ptime = max(0.01, ptime)

        lot.setup_req = setup_req

        if self.is_batch:
            yield self.env.process(self._batch_proc(lot, actual_ptime))
        else:
            yield self.env.process(self._single_proc(lot, actual_ptime, setup_req, setup_cost))

    def _single_proc(self, lot, ptime: float, setup_req: str, setup_cost: float):
        if self.dispatcher:
            self.waiting_lots.append(lot)
            scores = self.dispatcher.compute_scores(self, self.waiting_lots)
            score = scores.get(lot.id, 0)
            gnn_priority = 1000 - score
            urgency = lot.get_cqt_urgency(self.env.now, self.mean_ptime)
            log_msg = f"GNN Scoring: Lot {lot.id} -> Score: {score:.1f} (CQT Urgency: {urgency:.2f})"
            self.action_logs.append(log_msg)

            with self.res.request(priority=gnn_priority) as req:
                yield req
                if lot in self.waiting_lots:
                    self.waiting_lots.remove(lot)
                if setup_req and setup_req != self.current_setup and setup_cost > 0:
                    self.stats["setup_time"] += setup_cost
                    yield self.env.timeout(setup_cost)
                    self.current_setup = setup_req
                while self.is_down:
                    yield self.env.timeout(30)
                start = self.env.now
                lot.current_station = self.name
                yield self.env.timeout(ptime)
                self.stats["util_time"] += self.env.now - start
                self.stats["lots_processed"] += 1
        else:
            with self.res.request() as req:
                yield req
                if setup_req and setup_req != self.current_setup and setup_cost > 0:
                    self.stats["setup_time"] += setup_cost
                    yield self.env.timeout(setup_cost)
                    self.current_setup = setup_req
                while self.is_down:
                    yield self.env.timeout(30)
                start = self.env.now
                lot.current_station = self.name
                yield self.env.timeout(ptime)
                self.stats["util_time"] += self.env.now - start
                self.stats["lots_processed"] += 1

    def _batch_proc(self, lot, ptime: float):
        self.batch_queue.append(lot)
        lot.wait_event = self.env.event()
        lot.batch_done_ev = None
        is_leader = False

        if self.queued_wafers >= self.batch_min_wafers:
            self._release_batch_with_event(lot)
        else:
            timeout_ev = self.env.timeout(self.BATCH_WAIT_MAX)
            yield lot.wait_event | timeout_ev
            if not lot.wait_event.triggered:
                self._release_batch_with_event(lot)

        is_leader = getattr(lot, 'is_leader', False)
        lot.current_station = self.name

        if is_leader:
            with self.res.request() as req:
                yield req
                while self.is_down:
                    yield self.env.timeout(30)
                start = self.env.now
                yield self.env.timeout(ptime)
                self.stats["util_time"] += self.env.now - start
                self.stats["lots_processed"] += 1
            if lot.batch_done_ev and not lot.batch_done_ev.triggered:
                lot.batch_done_ev.succeed()
        else:
            if lot.batch_done_ev and not lot.batch_done_ev.triggered:
                yield lot.batch_done_ev
            else:
                yield self.env.timeout(0)
            self.stats["lots_processed"] += 1

    def _release_batch_with_event(self, trigger_lot=None):
        waiting = list(self.batch_queue)
        self.batch_queue = []
        if not waiting:
            return
        done_ev = self.env.event()
        leader = waiting[0]
        leader.is_leader = True
        for l in waiting:
            l.batch_done_ev = done_ev
            if l is not leader:
                l.is_leader = False
                if l.wait_event and not l.wait_event.triggered:
                    l.wait_event.succeed()
        if leader.wait_event and not leader.wait_event.triggered:
            leader.wait_event.succeed()


def failure_process(env, station: AdvancedStation, mttf: float, mttr: float):
    while True:
        yield env.timeout(random.expovariate(1.0 / mttf))
        station.is_down = True
        station.last_failure_time = env.now
        down_dur = random.expovariate(1.0 / mttr)
        station.stats["down_time"] += down_dur
        yield env.timeout(down_dur)
        station.is_down = False
        station.last_recovery_time = env.now
