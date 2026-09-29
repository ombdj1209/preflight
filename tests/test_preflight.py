import numpy as np
import pytest

from preflight.recommend import Rec, RecommendationService, apply, still_valid
from preflight.sandbox import evaluate, hindsight, horizon
from preflight.sim import Config, Sim

END = 6 * 3600


def run(seed, t=END):
    s = Sim(seed=seed)
    s.run_until(t)
    return s


def test_run_is_deterministic():
    assert run(7).breakdown() == run(7).breakdown()


def test_every_tote_is_delivered_exactly_once():
    s = run(3)
    assert s.m.totes_delivered == s.total_totes
    assert s.conv_n == 0 and not s.queue and not s.held


def test_fork_without_action_reproduces_original_future():
    base = Sim(seed=5)
    base.run_until(4000)
    f = base.fork()
    base.run_until(END)
    f.run_until(END)
    assert f.breakdown() == pytest.approx(base.breakdown())


def test_common_random_numbers_give_zero_delta_for_identical_branches():
    s = run(9, 4000)
    a, b = s.fork(123), s.fork(123)
    a.run_until(8000)
    b.run_until(8000)
    assert a.cost() == pytest.approx(b.cost())


def test_fork_is_isolated_from_parent():
    s = run(4, 3000)
    before = s.breakdown()
    f = s.fork()
    f.release_batch(next(iter(f.batches)))
    f.run_until(9000)
    assert s.breakdown() == before


def test_delay_truck_invalidates_the_original_departure():
    s = Sim(seed=1)
    tr = s.trucks[0]
    s.run_until(tr.depart_at - 10)
    assert s.delay_truck(0, 600)
    s.run_until(tr.depart_at + 1)
    assert not s.trucks[0].departed
    s.run_until(tr.depart_at + 601)
    assert s.trucks[0].departed


def test_cancel_item_releases_a_held_tote():
    for seed in range(40):
        s = Sim(seed=seed)
        s.run_until(5000)
        if s.held:
            oid = next(iter(s.held))
            assert s.cancel_item(oid)
            assert oid not in s.held
            return
    pytest.skip("no held tote found")


def test_race_guard_rejects_stale_recommendation():
    s = Sim(seed=2)
    s.run_until(600)
    bid = next(iter(s.batches))
    rec = Rec("release_batch", f"batch:{bid}", bid, s.batches[bid].truck, 1000.0, 2)
    s.release_batch(bid)                     # someone else already did it
    svc = RecommendationService()
    assert not still_valid(s, rec)
    assert svc.accept(s, rec) is False
    assert svc.log[-1]["decision"] == "stale"


def test_sql_models_emit_known_types():
    s, svc = Sim(seed=2), RecommendationService()
    seen = set()
    for t in range(60, 3 * 3600, 120):
        s.run_until(t)
        seen |= {r.rec_type for r in svc.refresh(s)}
    assert seen <= {"release_batch", "cancel_item", "delay_truck"} and seen


def test_evaluate_is_reproducible_and_horizon_covers_truck():
    s, svc = Sim(seed=3), RecommendationService()
    s.run_until(4000)
    rec = svc.refresh(s)[0]
    assert horizon(s, rec) >= s.trucks[rec.truck_id].depart_at
    a, b = evaluate(s, rec, n_seeds=4), evaluate(s, rec, n_seeds=4)
    assert a.delta == b.delta and a.ci_low <= a.delta <= a.ci_high


def test_hindsight_does_not_mutate_decision_state():
    s, svc = Sim(seed=3), RecommendationService()
    s.run_until(4000)
    rec = svc.refresh(s)[0]
    before = s.breakdown()
    hindsight(s, rec)
    assert s.breakdown() == before
