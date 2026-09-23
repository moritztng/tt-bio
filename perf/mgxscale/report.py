#!/usr/bin/env python3
"""Turn the throughput JSONL into the table and the campaign arithmetic.

The arithmetic is the point. A rate in designs per hour per chip only answers the row's
question once it is carried to "chip-hours for one target's design funnel" and then to "can
27 chips cover five targets in the 34-day window", which is what a credit sponsorship would
be sized against.

    python3 perf/mgxscale/report.py perf/mgxscale/results/*.jsonl

A row that did not reach the model is REFUSED here rather than averaged in, on the same rule
`perf/mgxdesign/report.py` applies: a job killed by contention measured the fleet, not the
model. A PARTIAL (fewer designs on disk than asked) is kept and flagged, because the gap
between asked and written IS the finding about the batch axis.
"""
import argparse
import json
import pathlib
import sys

# BoltzGen's own shipped production default, and what it keeps. `tt_bio/main.py` documents
# --num_designs as "10000 in total (boltzgen)" and --budget as "30" kept after filtering.
BG_DEFAULT_DESIGNS = 10000
BG_DEFAULT_BUDGET = 30
WINDOW_DAYS = 34          # Adaptyv design window, 28 Sep - 31 Oct 2026
PROBLEMS = 5
USABLE_CHIPS = 27         # 32 on whglx less the five cardblocked (1, 24-27)


def load(paths) -> list[dict]:
    rows = []
    for p in paths:
        for line in pathlib.Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fmt_clock(r) -> str:
    c = r.get("aiclk") or {}
    return f"{c.get('median', '?')} (n={c.get('n', 0)})"


def fmt_load(r) -> str:
    l = r.get("load") or {}
    return f"{l.get('median', '?')}/{l.get('nproc', '?')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--tag", default=None, help="only rows with this tag")
    args = ap.parse_args()
    rows = load(args.jsonl)
    if args.tag:
        rows = [r for r in rows if r.get("tag") == args.tag]
    if not rows:
        print("no rows")
        return 1

    print(f"{'model':9} {'tgt':>5} {'ask':>4} {'got':>4} {'steps':>5} {'wall_s':>8} "
          f"{'s/design':>9} {'des/h':>7} {'AICLK med':>12} {'load':>9} {'card':>4} verdict")
    for r in sorted(rows, key=lambda r: (r["model"], r["target_res"], r["asked"])):
        print(f"{r['model']:9} {r['target_res']:>5} {r['asked']:>4} {r.get('n_designs', 0):>4} "
              f"{r.get('steps', '-'):>5} {r.get('wall_s', 0):>8.1f} "
              f"{r.get('s_per_design', float('nan')):>9.1f} "
              f"{r.get('designs_per_h', float('nan')):>7.2f} {fmt_clock(r):>12} "
              f"{fmt_load(r):>9} {r.get('card', '-'):>4} "
              f"{r['verdict']}{'' if r['verdict'] == 'PASS' else ' ' + r.get('mechanism', '')}")

    print("\n-- campaign arithmetic, per model, from the best measured rate --")
    best: dict[str, dict] = {}
    for r in rows:
        if r["verdict"] == "FAIL" or not r.get("designs_per_h"):
            continue
        k = (r["model"], r["target_res"])
        if k not in best or r["designs_per_h"] > best[k]["designs_per_h"]:
            best[k] = r
    for (model, tgt), r in sorted(best.items()):
        rate = r["designs_per_h"]
        chip_h_10k = BG_DEFAULT_DESIGNS / rate
        per_window = rate * 24 * WINDOW_DAYS * USABLE_CHIPS
        print(f"{model} @ {tgt} res: {rate:.2f} designs/h/chip at n={r['n_designs']} per job\n"
              f"    {BG_DEFAULT_DESIGNS} designs (boltzgen's shipped default) = "
              f"{chip_h_10k:.0f} chip-hours = {chip_h_10k / 24:.1f} chip-days "
              f"= {chip_h_10k / USABLE_CHIPS:.1f} h on {USABLE_CHIPS} chips\n"
              f"    {USABLE_CHIPS} chips x {WINDOW_DAYS} d = {per_window:,.0f} designs, "
              f"{per_window / PROBLEMS:,.0f} per problem across {PROBLEMS} problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
