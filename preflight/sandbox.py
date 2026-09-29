"""Counterfactual sandbox (Phase 2), learned impact predictor (Phase 1) and the feedback
grader (the precondition: was the decision right, in hindsight?).

A recommendation is evaluated as a paired experiment: fork the live state twice with the
same noise seed, apply the action in one branch only, run both to a horizon that covers the
affected truck, and compare cost. Common random numbers make the paired delta low-variance.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .recommend import DELAY_TRUCK_S, Rec, RecommendationService, apply
from .sim import Config, Sim, Weights

TYPES = ("release_batch", "cancel_item", "delay_truck")


def horizon(sim: Sim, rec: Rec) -> float:
    tr = sim.trucks[rec.truck_id]
    extra = DELAY_TRUCK_S if rec.rec_type == "delay_truck" else 0.0
    end = tr.depart_at + tr.delay_s + extra + sim.cfg.conveyor_transit_s + 900.0
    return max(sim.t + 1800.0, end)


@dataclass
class Impact:
    delta: float            # cost(accept) - cost(reject); negative means accept is better
    ci_low: float
    ci_high: float
    p_accept_better: float
    components: dict        # mean paired delta per outcome component
    n: int
    elapsed_ms: float

    @property
    def accept(self) -> bool:
        return self.delta < 0


def paired_runs(sim: Sim, rec: Rec, n_seeds: int, sigma: float, seed0: int):
    h = horizon(sim, rec)
    out = []
    for k in range(n_seeds):
        ns = seed0 + k
        a, b = sim.fork(ns, sigma), sim.fork(ns, sigma)
        apply(a, rec)
        a.run_until(h)
        b.run_until(h)
        out.append((a.breakdown(), b.breakdown()))
    return out


def evaluate(sim: Sim, rec: Rec, w: Weights = Weights(), n_seeds: int = 8, sigma: float = 0.25,
             seed0: int = 10_000) -> Impact:
    t0 = time.perf_counter()
    runs = paired_runs(sim, rec, n_seeds, sigma, seed0)
    d = np.array([Sim.cost_of(a, w) - Sim.cost_of(b, w) for a, b in runs])
    comps = {k: float(np.mean([a[k] - b[k] for a, b in runs]))
             for k in ("late_orders", "late_minutes", "incomplete", "blocked_minutes")}
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    return Impact(float(d.mean()), float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se),
                  float((d < 0).mean()), comps, len(d), (time.perf_counter() - t0) * 1e3)


def hindsight(sim_at_decision: Sim, rec: Rec, w: Weights = Weights()) -> tuple[float, float]:
    """Cost of (accept, reject) under the *true* future (no model noise)."""
    h = horizon(sim_at_decision, rec)
    a, b = sim_at_decision.fork(), sim_at_decision.fork()
    apply(a, rec)
    a.run_until(h)
    b.run_until(h)
    return a.cost(w), b.cost(w)


def break_even_incomplete(impact: Impact, w: Weights = Weights()) -> float | None:
    """For a cancel_item decision: the incomplete-order weight at which cancel and wait tie.
    Makes the late-vs-incomplete trade-off explicit instead of hiding it in a constant."""
    c = impact.components
    if abs(c["incomplete"]) < 1e-9:
        return None
    rest = (w.late_order * c["late_orders"] + w.late_minute * c["late_minutes"]
            + w.blocked_minute * c["blocked_minutes"])
    return -rest / c["incomplete"]


# --------------------------------------------------------------------- Phase 1 predictor
def features(sim: Sim, rec: Rec) -> list[float]:
    tr = sim.trucks[rec.truck_id]
    live = [o for o in sim.orders.values() if not o.done and o.truck == rec.truck_id]
    missing = sum(o.n_totes - o.arrived for o in live)
    o = sim.orders.get(rec.target) if rec.rec_type == "cancel_item" else None
    return [
        *(1.0 if rec.rec_type == t else 0.0 for t in TYPES),
        rec.seconds_to_departure / 3600.0,
        sim.conv_n / sim.cfg.conveyor_capacity,
        len(sim.queue) / (sim.cfg.n_stations * 10.0),
        sum(1 for b in sim.st_busy if not b) / sim.cfg.n_stations,
        len(sim.blocked) / sim.cfg.n_stations,
        missing / 50.0,
        ((o.replenish_at - sim.t) / 3600.0) if o else 0.0,
        float(np.mean([x.slack for x in live])) / 900.0 if live else 0.0,
        tr.delay_s / 600.0,
    ]


class ImpactModel:
    def __init__(self):
        self.m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                           subsample=0.8, random_state=0)

    def fit(self, X, y):
        self.m.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, sim: Sim, rec: Rec) -> float:
        return float(self.m.predict(np.asarray([features(sim, rec)]))[0])


# ------------------------------------------------------------------------- shift runner
def run_shift(seed: int, controller: str, cfg: Config = Config(), w: Weights = Weights(),
              model: ImpactModel | None = None, n_seeds: int = 8, epoch_s: float = 60.0,
              max_recs_per_epoch: int = 3, collect: list | None = None) -> dict:
    """controller: 'automation_only' | 'accept_all' | 'sandbox' | 'predictor'."""
    sim, svc = Sim(cfg, seed), RecommendationService()
    t, decide_ms = 0.0, []
    while t < cfg.shift_s:
        t += epoch_s
        sim.run_until(t)
        if controller == "automation_only":
            continue
        for rec in svc.refresh(sim)[:max_recs_per_epoch]:
            if collect is not None:
                collect.append((sim.fork(), rec))
            if controller == "accept_all":
                svc.accept(sim, rec)
                continue
            t0 = time.perf_counter()
            if controller == "sandbox":
                good = evaluate(sim, rec, w, n_seeds=n_seeds).accept
            elif controller == "predictor":
                good = model.predict(sim, rec) < 0
            else:
                raise ValueError(controller)
            decide_ms.append((time.perf_counter() - t0) * 1e3)
            svc.accept(sim, rec) if good else svc.reject(sim, rec)
    sim.run_until(cfg.shift_s + 2 * 3600)
    b = sim.breakdown()
    return {"controller": controller, "seed": seed, "cost": sim.cost(w), **b,
            "decisions": len(svc.log),
            "accepted": sum(1 for x in svc.log if x["decision"] == "accept"),
            "decide_ms_p50": float(np.percentile(decide_ms, 50)) if decide_ms else 0.0,
            "decide_ms_p95": float(np.percentile(decide_ms, 95)) if decide_ms else 0.0,
            "totes_delivered": sim.m.totes_delivered, "total_totes": sim.total_totes}
