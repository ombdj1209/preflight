"""Reproduces every number in the README. Runtime: a few minutes on one CPU core.

    python run_demo.py            # full run
    python run_demo.py --quick    # smaller sample, for CI
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from preflight.report import write_report
from preflight.sandbox import (ImpactModel, break_even_incomplete, evaluate, features, hindsight,
                               run_shift)
from preflight.sim import Weights

OUT = Path(__file__).resolve().parent / "results"


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return {"mean": float(x.mean()), "ci95": float(1.96 * se), "n": int(len(x))}


def main(quick: bool):
    w = Weights()
    t_start = time.time()
    train_seeds = range(100, 106 if quick else 114)
    test_seeds = range(1, 4 if quick else 9)

    # 1. collect decision points from rule-following shifts, label them with the sandbox
    print("collecting training decisions ...")
    train = []
    for s in train_seeds:
        run_shift(s, "accept_all", collect=train)
    X = [features(sim, rec) for sim, rec in train]
    y = [evaluate(sim, rec, w, n_seeds=16, seed0=50_000).delta for sim, rec in train]
    model = ImpactModel().fit(X, y)

    # 2. held-out decisions: fidelity/speed study against a K=64 reference and against the truth
    print("fidelity study ...")
    held = []
    for s in test_seeds:
        run_shift(s, "accept_all", collect=held)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(held), size=min(len(held), 40 if quick else 100), replace=False)
    sample = [held[i] for i in idx]
    ref = [evaluate(sim, rec, w, n_seeds=64, seed0=90_000) for sim, rec in sample]
    truth = [hindsight(sim, rec, w) for sim, rec in sample]
    truth_accept = [a < b for a, b in truth]
    fidelity = []
    for k in (1, 2, 4, 8, 16, 32):
        ims = [evaluate(sim, rec, w, n_seeds=k, seed0=70_000) for sim, rec in sample]
        fidelity.append({
            "method": f"sandbox K={k}",
            "agree_with_reference": float(np.mean([a.accept == r.accept for a, r in zip(ims, ref)])),
            "agree_with_truth": float(np.mean([a.accept == t for a, t in zip(ims, truth_accept)])),
            "ms_p50": float(np.median([a.elapsed_ms for a in ims])),
            "ms_p95": float(np.percentile([a.elapsed_ms for a in ims], 95))})
    t0 = time.perf_counter()
    preds = [model.predict(sim, rec) for sim, rec in sample]
    pred_ms = (time.perf_counter() - t0) * 1e3 / len(sample)
    fidelity.append({
        "method": "learned predictor",
        "agree_with_reference": float(np.mean([(p < 0) == r.accept for p, r in zip(preds, ref)])),
        "agree_with_truth": float(np.mean([(p < 0) == t for p, t in zip(preds, truth_accept)])),
        "ms_p50": pred_ms, "ms_p95": pred_ms,
        "mae_vs_reference": float(np.mean([abs(p - r.delta) for p, r in zip(preds, ref)]))})
    fidelity.append({
        "method": "rules only (always accept)",
        "agree_with_reference": float(np.mean([r.accept for r in ref])),
        "agree_with_truth": float(np.mean(truth_accept)), "ms_p50": 0.0, "ms_p95": 0.0})

    # 3. feedback grader: how often was the rule-following decision right in hindsight?
    regrets = [max(0.0, a - b) for a, b in truth]
    grader = {"decisions": len(truth), "accept_was_right": float(np.mean(truth_accept)),
              "total_regret": float(np.sum(regrets)),
              "by_type": {}}
    for t in ("release_batch", "cancel_item", "delay_truck"):
        sel = [i for i, (_, rec) in enumerate(sample) if rec.rec_type == t]
        if sel:
            grader["by_type"][t] = {"n": len(sel),
                                    "accept_was_right": float(np.mean([truth_accept[i] for i in sel])),
                                    "regret": float(np.sum([regrets[i] for i in sel]))}

    # 4. late vs incomplete: where does the cancel decision flip?
    be = [break_even_incomplete(r, w) for (sim, rec), r in zip(sample, ref) if rec.rec_type == "cancel_item"]
    be = [x for x in be if x is not None and np.isfinite(x)]
    break_even = {"n": len(be), "p25": float(np.percentile(be, 25)) if be else None,
                  "median": float(np.median(be)) if be else None,
                  "p75": float(np.percentile(be, 75)) if be else None,
                  "current_weight": w.incomplete}

    # 5. end-to-end shifts on held-out seeds
    print("shift comparison ...")
    shifts = []
    for s in test_seeds:
        for c in ("automation_only", "accept_all", "sandbox", "predictor"):
            shifts.append(run_shift(s, c, w=w, model=model, n_seeds=8))
    by = {c: [r for r in shifts if r["controller"] == c] for c in
          ("automation_only", "accept_all", "sandbox", "predictor")}
    base = {r["seed"]: r["cost"] for r in by["accept_all"]}
    summary = {}
    for c, rows in by.items():
        summary[c] = {"cost": mean_ci([r["cost"] for r in rows]),
                      "late_orders": mean_ci([r["late_orders"] for r in rows]),
                      "incomplete": mean_ci([r["incomplete"] for r in rows]),
                      "blocked_minutes": mean_ci([r["blocked_minutes"] for r in rows]),
                      "cost_change_vs_rules_pct": mean_ci(
                          [100 * (r["cost"] - base[r["seed"]]) / base[r["seed"]] for r in rows]),
                      "decide_ms_p95": float(np.max([r["decide_ms_p95"] for r in rows]))}

    results = {"environment": {"python": platform.python_version(), "machine": platform.machine(),
                               "processor": platform.processor() or "unknown",
                               "runtime_s": round(time.time() - t_start, 1), "quick": quick},
               "weights": w.__dict__, "training_decisions": len(train),
               "fidelity": fidelity, "grader": grader, "break_even_incomplete": break_even,
               "shifts": summary, "shift_rows": shifts}
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    write_report(results, OUT / "report.html")
    print(json.dumps({k: results[k] for k in ("fidelity", "grader", "break_even_incomplete")}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time() - t_start:.0f}s -> results/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
