# Preflight

**See the cascade before you click Accept.** A decision-impact sandbox for automated fulfilment centres: a discrete-event simulation of pick stations, a finite conveyor and dock departures; dbt-style SQL recommendation rules; a race-guarded recommendation service; paired counterfactual evaluation of every recommendation; a learned impact predictor; and a hindsight grader that scores past decisions.

All data is synthetic, seeded and reproducible. Nothing here uses or claims knowledge of Picnic's internal systems or data.

## Why this exists

Picnic's engineering blog (*Our vision of building an Intelligent Control Center for Fulfilment*, July 2026) describes a control room where the system suggests actions and controllers accept or reject them, built on dbt models over ClickHouse streaming to RabbitMQ. It names the open problem plainly: a controller cannot see whether a decision was right, or what it will cause downstream. The published roadmap is a feedback loop first, then predictive models, then a discrete-event simulation that tests a decision before it happens, then reinforcement learning. It also asks how to weigh a late delivery against an incomplete one.

Preflight is a small, public, working version of that roadmap, built to answer one question the roadmap has to settle: **how much simulation fidelity is enough to trust a sandbox verdict, and can it run fast enough to show before the click?**

## What it does

| Piece | File | Mirrors |
|---|---|---|
| Seeded FC simulation, cheap exact forks | `preflight/sim.py` | Phase 2 digital twin |
| Recommendation rules as SQL models | `models/*.sql` | dbt models on the real-time store |
| Snapshot tables, prioritised queue, accept/reject log, live-state re-check on Accept | `preflight/recommend.py` | Recommendation service and its race guard |
| Paired counterfactual evaluation with common random numbers | `preflight/sandbox.py` `evaluate` | "Test the decision seconds before it happens" |
| Gradient-boosted impact predictor trained on sandbox labels | `preflight/sandbox.py` `ImpactModel` | Phase 1 predictive model |
| Hindsight grader (true future, both branches) | `preflight/sandbox.py` `hindsight` | Precondition: the feedback loop |
| Break-even weight for late versus incomplete | `preflight/sandbox.py` `break_even_incomplete` | The open optimisation question |

The recommendation types are: release the next batch early, cancel a missing item so the tote can leave, and hold a truck for ten minutes. If a controller rejects, the default automation still runs (batches release on schedule, totes wait for replenishment, trucks leave on time), so every recommendation is a clean A/B choice.

## Results

Produced by `python run_demo.py` in 230 s on one CPU core (Python 3.12). Held-out days are simulation seeds never used for training.

### How much simulation is enough?

100 held-out decisions. Each method's accept/reject call is compared with a K=64 sandbox reference and with the true outcome (both branches replayed with the real future).

| Method | Agrees with K=64 | Agrees with true outcome | p50 latency |
|---|---|---|---|
| Rules only (always accept) | 46% | 42% | 0 ms |
| Learned predictor | 86% | 82% | 0.27 ms |
| Sandbox K=1 | 94% | 92% | 5.4 ms |
| Sandbox K=2 | 98% | 96% | 10.8 ms |
| Sandbox K=4 | 99% | 97% | 21 ms |
| Sandbox K=8 | 99% | 95% | 42 ms |
| Sandbox K=32 | 99% | 95% | 167 ms |

**Reading it:** agreement with the reference saturates at K=4, in about 21 ms. Beyond that, extra runs buy nothing; the remaining gap to the truth is model error (the sandbox does not know the real pick times), not sampling noise. So the next engineering hour belongs in simulation fidelity, not in more runs. The learned predictor is about 80 times faster than K=4 but wrong on 18% of decisions against the truth versus 3%, which suits pre-filtering rather than final verdicts.

### Were the rule-based recommendations right?

Hindsight grading of the same 100 decisions: accepting was the better choice 42% of the time.

| Type | n | Accepting was right | Regret (cost units) |
|---|---|---|---|
| Release batch early | 80 | 29% | 7.0 |
| Cancel missing item | 20 | 95% | 25.0 |

Early-release suggestions are mostly noise: they rarely change which truck an order makes and they add conveyor blocking. Cancel suggestions are nearly always right. That split is exactly the signal the blog describes wanting: always-accepted types can be automated, mostly-rejected types need better rules.

### Late versus incomplete

Instead of hard-coding the answer, Preflight computes where the decision flips. Across 20 cancel-or-wait decisions, cancelling beats waiting whenever one incomplete order costs less than a median of **40** cost units (interquartile 37 to 41). The configured weight is 25, so the policy cancels. A business owner can now argue about one number with its consequences visible.

### End-to-end shifts

Eight held-out simulated shifts, same day four ways. Change is paired per day against following every recommendation; ± is a 95% interval.

| Controller | Mean shift cost | Change vs rules | Late orders | Incomplete | Blocked station-min | Decision p95 |
|---|---|---|---|---|---|---|
| Automation only (ignore feed) | 200.0 ± 34.5 | +42.8% | 4.6 | 0.0 | 254 | n/a |
| Accept every recommendation | 138.7 ± 18.5 | baseline | 0.0 | 5.0 | 273 | n/a |
| Sandbox K=8 decides | 127.8 ± 22.5 | −8.8% ± 7.9 | 0.0 | 4.6 | 243 | 55 ms |
| Learned predictor decides | 127.7 ± 22.5 | −8.9% ± 7.9 | 0.0 | 4.6 | 242 | 1 ms |

The recommendation feed itself is worth about 30% over pure automation. Screening each recommendation removes another 9% on top, mostly by rejecting early releases that only add conveyor blocking. The interval is wide with eight days; the direction is consistent.

## Run it

```bash
pip install -e ".[dev]"
python -m pytest -q          # 11 tests: determinism, conservation, fork isolation, CRN, race guard, SQL models
python run_demo.py           # full benchmark, about 4 minutes on one core
python run_demo.py --quick   # smaller sample for CI
open results/report.html
```

## Design notes

- **Forks copy only live state.** Finished orders and released batches are not copied, so a fork costs microseconds and a 30 to 60 minute counterfactual costs about 5 ms.
- **Common random numbers.** Both branches of a paired run share the same noise draw, so the delta measures the decision, not the dice. A test proves identical branches give a delta of exactly zero.
- **Every stochastic input is pre-sampled.** A run is a pure function of (config, seed, decisions), so any past shift can be replayed decision by decision.
- **The horizon covers the affected truck.** Evaluation runs until the relevant departure plus transit, so waiting is never favoured just because its cost lands after the horizon. Orders still stranded at the horizon are charged their known lower-bound lateness.
- **Race guard.** On Accept, the service re-checks live state (batch not already released, item not already replenished, truck not gone) and records a stale decision instead of acting.

## Stack mapping

| Here | Production equivalent |
|---|---|
| DuckDB over snapshot DataFrames | ClickHouse real-time store |
| `models/*.sql` | dbt models on a 30 to 60 s schedule |
| In-process `RecommendationService` | Recommendation service fed by the ClickHouse RabbitMQ engine |
| `Sim` | Discrete-event twin fed from warehouse events |

## Limitations

- The FC is a simplified single-stage model: one station type, one conveyor buffer, one dock. Real automated FCs have many interacting loops.
- Sandbox uncertainty is multiplicative noise on pick times only. Hardware faults, labour and inbound variability are not modelled.
- Each sandbox run evaluates one decision against default automation. It does not simulate the controller's future choices.
- The cost weights are illustrative. The point of the break-even analysis is to make them explicit, not to claim the right values.
- Results come from one CPU core. Absolute latencies will differ on other hardware; the relative ordering is what matters.

## Next steps

1. Calibrate the noise model against logged pick-time residuals, since model error, not sampling, now dominates.
2. Add truck-hold decisions to the benchmark sample (none occurred in the held-out draw).
3. Use the predictor as a pre-filter: run the sandbox only when the predicted delta is near zero.
4. Export the decision log to a dbt model so graded outcomes feed rule tuning, closing the loop.
