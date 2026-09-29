"""Discrete-event simulation of a synthetic automated fulfilment centre (FC).

Flow: batch release -> pick stations -> conveyor (finite buffer) -> dock -> truck departure.
Every stochastic input is pre-sampled at construction, so a run is a pure function of
(config, seed, decisions). Forks copy only live state, which makes counterfactual
"what if I accept?" runs cheap and exactly reproducible.
"""
from __future__ import annotations

import copy
import dataclasses
import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np

STATION_DONE, ARRIVE, DEPART, REPLENISH, RELEASE = range(5)
NOISE_SLOTS = 8  # totes per order addressable by the noise table (totes_max must be <= this)


@dataclass(frozen=True)
class Config:
    n_stations: int = 6
    conveyor_capacity: int = 20
    conveyor_transit_s: float = 150.0
    pick_mean_s: float = 30.0
    pick_cv: float = 0.35
    shift_s: float = 3 * 3600.0
    first_departure_s: float = 3000.0
    n_trucks: int = 12
    orders_per_truck: int = 26
    totes_min: int = 2
    totes_max: int = 5
    batch_size_orders: int = 13
    release_lead_s: float = 2400.0      # default automation releases a batch this long before its truck
    shortage_rate: float = 0.05
    rescue_delay_s: float = 2700.0      # an order that misses its truck goes on a rescue run
    slot_slack_max_s: float = 900.0     # downstream slack before a truck delay makes a customer late


@dataclass(frozen=True)
class Weights:
    """How the business weighs outcomes. Deliberately explicit and tunable."""
    late_order: float = 10.0      # per order that reaches the customer late
    late_minute: float = 0.5      # per order-minute of lateness
    incomplete: float = 25.0      # per order delivered with a cancelled item
    blocked_minute: float = 0.05  # per station-minute blocked by a full conveyor


class Order:
    __slots__ = ("id", "truck", "n_totes", "pick_s", "arrived", "short_idx", "short_ok",
                 "cancelled", "replenish_at", "slack", "missed_at", "done")

    def __init__(self, oid, truck, n_totes, pick_s, short_idx, replenish_at, slack):
        self.id, self.truck, self.n_totes, self.pick_s = oid, truck, n_totes, pick_s
        self.arrived, self.short_idx, self.short_ok, self.cancelled = 0, short_idx, short_idx < 0, False
        self.replenish_at, self.slack, self.missed_at, self.done = replenish_at, slack, None, False


class Truck:
    __slots__ = ("id", "depart_at", "orders", "departed", "delay_s", "version")

    def __init__(self, tid, depart_at, orders):
        self.id, self.depart_at, self.orders = tid, depart_at, orders
        self.departed, self.delay_s, self.version = False, 0.0, 0


class Batch:
    __slots__ = ("id", "truck", "orders", "released", "due_at")

    def __init__(self, bid, truck, orders, due_at):
        self.id, self.truck, self.orders, self.released, self.due_at = bid, truck, orders, False, due_at


@dataclass
class Metrics:
    ontime: int = 0
    late_orders: int = 0
    late_s: float = 0.0
    incomplete: int = 0
    blocked_s: float = 0.0
    totes_delivered: int = 0


class Sim:
    def __init__(self, cfg: Config = Config(), seed: int = 0):
        assert cfg.totes_max <= NOISE_SLOTS
        self.cfg, self.seed = cfg, seed
        rng = np.random.default_rng(seed)
        self.t, self.heap, self.seq = 0.0, [], 0
        self.queue: deque = deque()
        self.held: dict = {}
        n = cfg.n_stations
        self.st_busy, self.st_tote, self.st_blocked_since = [False] * n, [None] * n, [None] * n
        self.blocked: deque = deque()
        self.conv_n, self.noise, self.m = 0, None, Metrics()
        self.orders: dict[int, Order] = {}
        self.trucks: list[Truck] = []
        self.batches: dict[int, Batch] = {}
        shape = 1.0 / cfg.pick_cv ** 2
        gap = (cfg.shift_s - cfg.first_departure_s) / max(1, cfg.n_trucks - 1)
        oid = bid = 0
        for tid in range(cfg.n_trucks):
            depart = cfg.first_departure_s + tid * gap
            ids = []
            for _ in range(cfg.orders_per_truck):
                nt = int(rng.integers(cfg.totes_min, cfg.totes_max + 1))
                pick = rng.gamma(shape, cfg.pick_mean_s / shape, size=nt)
                short = int(rng.integers(nt)) if rng.random() < cfg.shortage_rate else -1
                repl = float(max(60.0, depart + rng.uniform(-3000, 1500))) if short >= 0 else 0.0
                self.orders[oid] = Order(oid, tid, nt, pick, short, repl,
                                         float(rng.uniform(0, cfg.slot_slack_max_s)))
                if short >= 0:
                    self._push(repl, REPLENISH, oid)
                ids.append(oid)
                oid += 1
            self.trucks.append(Truck(tid, depart, tuple(ids)))
            self._push(depart, DEPART, tid, 0)
            for i in range(0, len(ids), cfg.batch_size_orders):
                due = max(0.0, depart - cfg.release_lead_s)
                self.batches[bid] = Batch(bid, tid, tuple(ids[i:i + cfg.batch_size_orders]), due)
                self._push(due, RELEASE, bid)
                bid += 1
        self.n_noise = oid * NOISE_SLOTS
        self.total_totes = sum(o.n_totes for o in self.orders.values())

    # ---------------------------------------------------------------- engine
    def _push(self, t, kind, a, b=0):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, kind, a, b))

    def run_until(self, t_end: float) -> None:
        heap = self.heap
        while heap and heap[0][0] <= t_end:
            t, _, kind, a, b = heapq.heappop(heap)
            self.t = t
            if kind == STATION_DONE:
                self._station_done(a)
            elif kind == ARRIVE:
                self._arrive(a, b)
            elif kind == DEPART:
                self._depart(a, b)
            elif kind == REPLENISH:
                self._replenish(a)
            else:
                self.release_batch(a)
        self.t = max(self.t, t_end)

    def _pick_time(self, oid, idx):
        base = self.orders[oid].pick_s[idx]
        if self.noise is not None:
            base *= self.noise[oid * NOISE_SLOTS + idx]
        return float(base)

    def _dispatch(self):
        for s in range(self.cfg.n_stations):
            if self.st_busy[s]:
                continue
            while self.queue:
                oid, idx = self.queue.popleft()
                o = self.orders.get(oid)
                if o is None:
                    continue
                if idx == o.short_idx and not o.short_ok and not o.cancelled:
                    self.held[oid] = (oid, idx)       # item missing: park tote, station moves on
                    continue
                self.st_busy[s], self.st_tote[s] = True, (oid, idx)
                self._push(self.t + self._pick_time(oid, idx), STATION_DONE, s)
                break
            if not self.queue:
                return

    def _station_done(self, s):
        if self.conv_n < self.cfg.conveyor_capacity:
            self._to_conveyor(self.st_tote[s])
            self.st_busy[s], self.st_tote[s] = False, None
            self._dispatch()
        else:
            self.st_blocked_since[s] = self.t
            self.blocked.append(s)

    def _to_conveyor(self, tote):
        self.conv_n += 1
        self._push(self.t + self.cfg.conveyor_transit_s, ARRIVE, tote[0], tote[1])

    def _arrive(self, oid, idx):
        self.conv_n -= 1
        self.m.totes_delivered += 1
        o = self.orders.get(oid)
        if o is not None and not o.done:
            o.arrived += 1
            if o.missed_at is not None and o.arrived == o.n_totes:
                self.m.late_orders += 1
                self.m.late_s += (self.t - o.missed_at) + self.cfg.rescue_delay_s
                self._finalize(o)
        if self.blocked:
            s = self.blocked.popleft()
            self.m.blocked_s += self.t - self.st_blocked_since[s]
            self.st_blocked_since[s] = None
            self._to_conveyor(self.st_tote[s])
            self.st_busy[s], self.st_tote[s] = False, None
            self._dispatch()

    def _depart(self, tid, version):
        tr = self.trucks[tid]
        if version != tr.version or tr.departed:
            return
        tr.departed = True
        for oid in tr.orders:
            o = self.orders.get(oid)
            if o is None or o.done:
                continue
            if o.arrived == o.n_totes:
                lateness = max(0.0, tr.delay_s - o.slack)
                if lateness > 0:
                    self.m.late_orders += 1
                    self.m.late_s += lateness
                else:
                    self.m.ontime += 1
                self._finalize(o)
            else:
                o.missed_at = self.t

    def _replenish(self, oid):
        o = self.orders.get(oid)
        if o is None or o.done:
            return
        o.short_ok = True
        if oid in self.held:
            self.queue.appendleft(self.held.pop(oid))
            self._dispatch()

    def _finalize(self, o):
        o.done = True
        if o.cancelled:
            self.m.incomplete += 1

    # ------------------------------------------------------------- decisions
    def release_batch(self, bid) -> bool:
        b = self.batches.get(bid)
        if b is None or b.released:
            return False
        b.released = True
        for oid in b.orders:
            o = self.orders.get(oid)
            if o is not None:
                self.queue.extend((oid, i) for i in range(o.n_totes))
        self._dispatch()
        return True

    def cancel_item(self, oid) -> bool:
        o = self.orders.get(oid)
        if o is None or o.done or o.short_ok or o.cancelled:
            return False
        o.cancelled = True
        if oid in self.held:
            self.queue.appendleft(self.held.pop(oid))
            self._dispatch()
        return True

    def delay_truck(self, tid, seconds) -> bool:
        tr = self.trucks[tid]
        if tr.departed:
            return False
        tr.delay_s += seconds
        tr.version += 1
        self._push(tr.depart_at + tr.delay_s, DEPART, tid, tr.version)
        return True

    # ---------------------------------------------------------------- output
    def breakdown(self) -> dict:
        late_n, late_s, incomplete = self.m.late_orders, self.m.late_s, self.m.incomplete
        for o in self.orders.values():
            if o.done:
                continue
            if o.missed_at is not None:  # horizon ends before the rescue run: charge the known lower bound
                late_n += 1
                late_s += (self.t - o.missed_at) + self.cfg.rescue_delay_s
            if o.cancelled:
                incomplete += 1
        blocked = self.m.blocked_s + sum(self.t - x for x in self.st_blocked_since if x is not None)
        return {"late_orders": late_n, "late_minutes": late_s / 60.0, "incomplete": incomplete,
                "blocked_minutes": blocked / 60.0, "ontime": self.m.ontime}

    @staticmethod
    def cost_of(b: dict, w: Weights) -> float:
        return (w.late_order * b["late_orders"] + w.late_minute * b["late_minutes"]
                + w.incomplete * b["incomplete"] + w.blocked_minute * b["blocked_minutes"])

    def cost(self, w: Weights = Weights()) -> float:
        return self.cost_of(self.breakdown(), w)

    def fork(self, noise_seed: int | None = None, sigma: float = 0.25) -> "Sim":
        """Copy live state. With noise_seed, future pick times get mean-one multiplicative noise.
        Two forks with the same noise_seed see identical randomness (common random numbers)."""
        s = Sim.__new__(Sim)
        s.__dict__.update(self.__dict__)
        s.heap, s.queue, s.held = list(self.heap), deque(self.queue), dict(self.held)
        s.st_busy, s.st_tote = list(self.st_busy), list(self.st_tote)
        s.st_blocked_since, s.blocked = list(self.st_blocked_since), deque(self.blocked)
        s.orders = {k: copy.copy(o) for k, o in self.orders.items() if not o.done}
        s.trucks = [copy.copy(t) for t in self.trucks]
        s.batches = {k: copy.copy(b) for k, b in self.batches.items() if not b.released}
        s.m = dataclasses.replace(self.m)
        if noise_seed is not None:
            z = np.random.default_rng(noise_seed).standard_normal(self.n_noise)
            s.noise = np.exp(sigma * z - 0.5 * sigma ** 2)
        return s
