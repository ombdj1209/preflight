"""Recommendation service: SQL models over a live snapshot, a prioritised queue, and a
race guard that re-validates live state at the moment a controller clicks Accept.

The SQL files in models/ play the role of dbt models; DuckDB stands in for the real-time
analytics store so the whole loop runs on a laptop."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from .sim import Sim

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DELAY_TRUCK_S = 600.0


@dataclass
class Rec:
    rec_type: str
    rec_key: str
    target: int
    truck_id: int
    seconds_to_departure: float
    severity: int
    created_t: float = 0.0

    @property
    def priority(self):
        return (-self.severity, self.seconds_to_departure)


def snapshot(sim: Sim) -> dict[str, pd.DataFrame]:
    t, cfg = sim.t, sim.cfg
    idle = sum(1 for b in sim.st_busy if not b)
    system = pd.DataFrame([{
        "t": t, "conveyor_n": sim.conv_n, "conveyor_capacity": cfg.conveyor_capacity,
        "conveyor_transit_s": cfg.conveyor_transit_s, "queue_len": len(sim.queue),
        "idle_stations": idle, "n_stations": cfg.n_stations, "held_n": len(sim.held)}])
    pending = [{"batch_id": b.id, "truck_id": b.truck,
                "n_totes": sum(sim.orders[o].n_totes for o in b.orders if o in sim.orders),
                "seconds_to_departure": sim.trucks[b.truck].depart_at + sim.trucks[b.truck].delay_s - t}
               for b in sim.batches.values() if not b.released and not sim.trucks[b.truck].departed]
    shortages, missing = [], {}
    for o in sim.orders.values():
        if o.done:
            continue
        tr = sim.trucks[o.truck]
        if not tr.departed:
            missing[o.truck] = missing.get(o.truck, 0) + (o.n_totes - o.arrived)
        if o.short_idx >= 0 and not o.short_ok and not o.cancelled and not tr.departed:
            shortages.append({"order_id": o.id, "truck_id": o.truck,
                              "seconds_to_departure": tr.depart_at + tr.delay_s - t,
                              "replenish_eta_s": o.replenish_at - t, "held": o.id in sim.held})
    trucks = [{"truck_id": tr.id, "seconds_to_departure": tr.depart_at + tr.delay_s - t,
               "totes_missing": missing.get(tr.id, 0), "delay_s": tr.delay_s}
              for tr in sim.trucks if not tr.departed]
    cols = {"pending_batches": ["batch_id", "truck_id", "n_totes", "seconds_to_departure"],
            "shortages": ["order_id", "truck_id", "seconds_to_departure", "replenish_eta_s", "held"],
            "trucks": ["truck_id", "seconds_to_departure", "totes_missing", "delay_s"]}
    frames = {"system": system}
    for name, rows in (("pending_batches", pending), ("shortages", shortages), ("trucks", trucks)):
        frames[name] = pd.DataFrame(rows, columns=cols[name]).astype(
            {"seconds_to_departure": "float64"}) if rows else pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in cols[name]})
    return frames


def apply(sim: Sim, rec: Rec) -> bool:
    if rec.rec_type == "release_batch":
        return sim.release_batch(rec.target)
    if rec.rec_type == "cancel_item":
        return sim.cancel_item(rec.target)
    if rec.rec_type == "delay_truck":
        return sim.delay_truck(rec.target, DELAY_TRUCK_S)
    raise ValueError(rec.rec_type)


def still_valid(sim: Sim, rec: Rec) -> bool:
    """Race guard: the snapshot may be seconds old; check the live core state before acting."""
    if rec.rec_type == "release_batch":
        b = sim.batches.get(rec.target)
        return b is not None and not b.released and not sim.trucks[b.truck].departed
    if rec.rec_type == "cancel_item":
        o = sim.orders.get(rec.target)
        return o is not None and not o.done and not o.short_ok and not o.cancelled
    if rec.rec_type == "delay_truck":
        return not sim.trucks[rec.target].departed
    return False


@dataclass
class RecommendationService:
    models_dir: Path = MODELS_DIR
    snooze_s: float = 300.0
    log: list = field(default_factory=list)
    _snoozed: dict = field(default_factory=dict)

    def __post_init__(self):
        self.models = {p.stem: p.read_text() for p in sorted(Path(self.models_dir).glob("*.sql"))}
        self.con = duckdb.connect()

    def refresh(self, sim: Sim) -> list[Rec]:
        frames = snapshot(sim)
        for name, df in frames.items():
            self.con.register(name, df)
        recs = []
        for sql in self.models.values():
            for row in self.con.execute(sql).fetchall():
                r = Rec(row[0], row[1], int(row[2]), int(row[3]), float(row[4]), int(row[5]), sim.t)
                if self._snoozed.get(r.rec_key, -1) > sim.t:
                    continue
                recs.append(r)
        return sorted(recs, key=lambda r: r.priority)

    def accept(self, sim: Sim, rec: Rec, **meta) -> bool:
        t0 = time.perf_counter()
        ok = still_valid(sim, rec) and apply(sim, rec)
        self.log.append({"t": sim.t, "rec_key": rec.rec_key, "rec_type": rec.rec_type,
                         "decision": "accept" if ok else "stale", "latency_us":
                         (time.perf_counter() - t0) * 1e6, **meta})
        return ok

    def reject(self, sim: Sim, rec: Rec, reason: str = "sandbox_predicts_worse", **meta) -> None:
        self._snoozed[rec.rec_key] = sim.t + self.snooze_s
        self.log.append({"t": sim.t, "rec_key": rec.rec_key, "rec_type": rec.rec_type,
                         "decision": "reject", "reason": reason, **meta})
