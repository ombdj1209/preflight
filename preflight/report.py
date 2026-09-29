"""Self-contained HTML report (no external assets) generated from results.json."""
from __future__ import annotations

import html
from pathlib import Path

CSS = """
:root{--ink:#1d2a24;--muted:#5b6b63;--line:#d7dfda;--bg:#fbfcfb;--accent:#2f7d57;--warn:#b5542c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 "Source Sans 3","Segoe UI",system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:48px 24px 80px}
h1{font-size:2.1rem;line-height:1.15;margin:0 0 8px;letter-spacing:-.01em}
h2{font-size:1.25rem;margin:48px 0 8px}p{max-width:68ch;color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:12px 0;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--muted)}td.n{text-align:right}th.n{text-align:right}
.bar{height:10px;background:var(--accent);border-radius:2px}.bar.w{background:var(--warn)}
.wrap{overflow-x:auto}
"""


def _tbl(headers, rows, numeric=()):
    h = "".join(f'<th class="{"n" if i in numeric else ""}">{html.escape(x)}</th>' for i, x in enumerate(headers))
    body = "".join("<tr>" + "".join(
        f'<td class="{"n" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def _bar(frac, warn=False):
    return f'<div class="bar{" w" if warn else ""}" style="width:{max(2, min(100, frac * 100)):.0f}%"></div>'


def write_report(r: dict, path: Path) -> None:
    fid = _tbl(["Method", "Agrees with K=64", "Agrees with true outcome", "p50 ms", "p95 ms"],
               [[html.escape(f["method"]), f'{f["agree_with_reference"]:.0%}', f'{f["agree_with_truth"]:.0%}',
                 f'{f["ms_p50"]:.2f}', f'{f["ms_p95"]:.2f}'] for f in r["fidelity"]], numeric=(1, 2, 3, 4))
    s = r["shifts"]
    worst = max(v["cost"]["mean"] for v in s.values())
    sh = _tbl(["Controller", "Shift cost", "", "Change vs rules", "Late orders", "Incomplete", "Blocked station-min"],
              [[c, f'{v["cost"]["mean"]:.1f} ± {v["cost"]["ci95"]:.1f}', _bar(v["cost"]["mean"] / worst),
                f'{v["cost_change_vs_rules_pct"]["mean"]:+.1f}%', f'{v["late_orders"]["mean"]:.1f}',
                f'{v["incomplete"]["mean"]:.1f}', f'{v["blocked_minutes"]["mean"]:.0f}'] for c, v in s.items()],
              numeric=(1, 3, 4, 5, 6))
    g = r["grader"]
    gr = _tbl(["Recommendation type", "Decisions", "Accepting was right", "Regret"],
              [[t, v["n"], f'{v["accept_was_right"]:.0%}', f'{v["regret"]:.1f}'] for t, v in g["by_type"].items()],
              numeric=(1, 2, 3))
    be = r["break_even_incomplete"]
    be_txt = ("No cancel decisions in the sample." if not be["n"] else
              f'Across {be["n"]} cancel-or-wait decisions, cancelling beats waiting whenever one incomplete order '
              f'costs less than a median of {be["median"]:.1f} cost units (interquartile {be["p25"]:.1f} to '
              f'{be["p75"]:.1f}). The current weight is {be["current_weight"]:.0f}.')
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Preflight results</title>
<style>{CSS}</style></head><body><main>
<h1>What happens if the controller clicks Accept?</h1>
<p>Synthetic fulfilment centre, seeded and reproducible. Every number on this page comes from
<code>run_demo.py</code> (runtime {r["environment"]["runtime_s"]} s, Python {r["environment"]["python"]}).</p>
<h2>Shift outcomes on held-out days</h2>
<p>Same simulated day, four ways of handling the recommendation feed.</p>{sh}
<h2>How much simulation is enough?</h2>
<p>Paired counterfactual runs per decision (K noise seeds, common random numbers) against a K=64
reference and against the true future.</p>{fid}
<h2>Were the rule-based recommendations right?</h2>
<p>Hindsight grading of {g["decisions"]} held-out decisions: accepting was right {g["accept_was_right"]:.0%}
of the time; total regret {g["total_regret"]:.1f} cost units.</p>{gr}
<h2>Late versus incomplete</h2><p>{be_txt}</p>
</main></body></html>"""
    Path(path).write_text(doc)
