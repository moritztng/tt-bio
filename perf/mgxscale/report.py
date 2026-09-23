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


def key(r) -> tuple:
    return (r["model"], r["target_res"], r["asked"], r.get("steps"), r.get("binder"),
            r.get("tag"))


def load(paths) -> list[dict]:
    """Rows, with a rescored row SUPERSEDING the run it corrects.

    A count that was wrong is not fixed by editing the JSONL -- the original row is what the
    harness saw and stays on disk. `job.py --rescore` re-derives the row from the artifacts,
    and the corrected one wins here, keyed on the job rather than on the file it came from."""
    rows = []
    for p in paths:
        for line in pathlib.Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    superseded = {key(r) for r in rows if r.get("rescored")}
    rows = [r for r in rows if r.get("rescored") or key(r) not in superseded]
    # Rows written before job.py learned to requeue a lost chip. The engine refused to open a
    # card another row had taken, so the job never reached the model: that measured the fleet.
    kept = [r for r in rows
            if not any("device contention" in d for d in (r.get("diag") or []))]
    if len(kept) != len(rows):
        print(f"[report] dropped {len(rows) - len(kept)} contention row(s): nothing ran")
    return kept


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

    # The binder column is not decoration. Without it the 1536 pxdesign refusal (a
    # 160-residue binder, 1696 tokens, over the recorded 1664-token wall) sits directly above
    # a 1536 PASS and reads as the same job failing and passing.
    print(f"{'model':9} {'tgt':>5} {'bndr':>5} {'ask':>4} {'got':>4} {'steps':>5} {'wall_s':>8} "
          f"{'s/design':>9} {'des/h':>7} {'AICLK med':>12} {'load':>9} {'card':>4} verdict")
    for r in sorted(rows, key=lambda r: (r["model"], r["target_res"], r["asked"])):
        spd = r.get("s_per_design")
        rate = r.get("designs_per_h")
        print(f"{r['model']:9} {r['target_res']:>5} {r.get('binder', '-'):>5} "
              f"{r['asked']:>4} {r.get('n_designs', 0):>4} "
              f"{r.get('steps', '-'):>5} {r.get('wall_s', 0):>8.1f} "
              f"{(f'{spd:.1f}' if spd else '-'):>9} {(f'{rate:.2f}' if rate else '-'):>7} "
              f"{fmt_clock(r):>12} {fmt_load(r):>9} {str(r.get('card', '-')):>4} "
              f"{r['verdict']}{'' if r['verdict'] == 'PASS' else ' ' + r.get('mechanism', '')}")

    print("\n-- campaign arithmetic, per model, from the best measured rate --")
    best: dict[str, dict] = {}
    for r in rows:
        if r["verdict"] == "FAIL" or not r.get("designs_per_h"):
            continue
        k = (r["model"], r["target_res"], r.get("binder"))
        if k not in best or r["designs_per_h"] > best[k]["designs_per_h"]:
            best[k] = r
    for (model, tgt, bndr), r in sorted(best.items()):
        rate = r["designs_per_h"]
        chip_h_10k = BG_DEFAULT_DESIGNS / rate
        per_window = rate * 24 * WINDOW_DAYS * USABLE_CHIPS
        print(f"{model} @ {tgt} res + {bndr} binder: {rate:.2f} designs/h/chip "
              f"at n={r['n_designs']} per job\n"
              f"    {BG_DEFAULT_DESIGNS} designs (boltzgen's shipped default) = "
              f"{chip_h_10k:.0f} chip-hours = {chip_h_10k / 24:.1f} chip-days "
              f"= {chip_h_10k / USABLE_CHIPS:.1f} h on {USABLE_CHIPS} chips\n"
              f"    {USABLE_CHIPS} chips x {WINDOW_DAYS} d = {per_window:,.0f} designs, "
              f"{per_window / PROBLEMS:,.0f} per problem across {PROBLEMS} problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
