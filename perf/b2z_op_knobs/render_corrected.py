#!/usr/bin/env python3
"""Render BEST-CONFIG-TABLE.md from the corrected (shipped-incumbent) sweep.

Two things the first render got wrong, both traced to the same straw man:

  * the RANKING. `baseline_wh_c1.json` timed every matmul as a bare `ttnn.linear(a, b)`, which
    costs 6-9x what the engine's own call costs, so matmuls were over-weighted by that much and
    the data-movement ops were pushed off the top of the table. Where the corrected sweep has an
    incumbent for an instance, its ms/fold is recomputed from that.
  * the GRID column, which measured the default resolver rather than anything reachable.

Non-matmul instances are not affected: their replay already passed the captured memory config and
compute kernel config, and ttnn has no core_grid argument for them. Their rows carry over.
"""
from __future__ import annotations

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "bench_manifest.json"))
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--resweep", required=True)
    ap.add_argument("--sweep", required=True, help="the first, straw-man sweep (non-matmul rows)")
    ap.add_argument("--renoise", required=True, help="the paired-repeat sweep, authoritative")
    ap.add_argument("--bitexact", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = {r["id"]: r for r in json.load(open(a.manifest))}
    base = {r["id"]: r for r in json.load(open(a.baseline))["rows"] if "ms_per_fold" in r}
    rs = json.load(open(a.resweep))
    rsi = {i["id"]: i for i in rs["instances"]}
    old = json.load(open(a.sweep))
    oldi = {i["id"]: i for i in old.get("instances", [])}
    rn = {i["id"]: i for i in json.load(open(a.renoise))["instances"]}
    bx = {r["id"]: r for r in json.load(open(a.bitexact))["rows"]} if a.bitexact else {}

    # corrected ms/fold
    corrected = {}
    for iid, r in base.items():
        calls = recs[iid]["calls_per_fold"]
        if iid in rsi and rsi[iid].get("incumbent_us"):
            corrected[iid] = (rsi[iid]["incumbent_us"] * calls / 1000.0, "shipped")
        else:
            corrected[iid] = (r["ms_per_fold"], "as-replayed")
    total = sum(v for v, _ in corrected.values())

    L = []
    w = L.append
    w("# BEST-CONFIG-TABLE — Boltz-2 512 aa, every knob on every instance above 0.5 % of device time")
    w("")
    w(f"whglx card 1, **Wormhole**, compute grid {rs['env']['grid']}, ttnn 0.68.0. Every number is a")
    w("paired interleaved ratio: incumbent, arm, incumbent, arm, ... in one process, each arm scored")
    w("against the median of the two incumbent runs that bracket it. `A/A` is the spread of the")
    w("incumbent runs themselves and is the noise floor a ratio has to clear — several instances here")
    w("have an A/A above 2, which means the box was contended while they ran and their arms say")
    w("nothing. `rel` is the arm's worst-element error against a float32 CPU reference computed from")
    w("the same seeded operands; the incumbent's own `rel` is printed beside it, because the shipped")
    w("config is not exact either and an arm is only losing accuracy if it is losing it *relative to")
    w("what ships*.")
    w("")
    w("**The incumbent is the call the fold makes.** Every matmul call site on the hot path already")
    w("passes `core_grid=CORE_GRID_MAIN` or a tuned program config, so a bare `ttnn.linear(a, b)`")
    w("incumbent measures ttnn's default resolver and nothing the fold can reach. The `nogrid=1` arm")
    w("below is that straw man, kept on purpose: it is 0.11x-0.16x on the pair-track Transition")
    w("matmuls, i.e. the default resolver is 6-9x slower than what already ships.")
    w("")
    w("## Corrected ranking")
    w("")
    w("The first ranking priced every matmul at the straw man's cost. Re-priced at the shipped cost:")
    w("")
    w("| # | instance | op | share | ms/fold | source | unit |")
    w("|---|---|---|---|---|---|---|")
    order = sorted(corrected.items(), key=lambda kv: -kv[1][0])
    for n, (iid, (ms, src)) in enumerate(order[:30], 1):
        rec = recs[iid]
        w(f"| {n} | `{iid}` | {rec['kind']} | {100*ms/total:.2f} % | {ms:.1f} | {src} | "
          f"{rec['unit_path'].rsplit('/',1)[-1]} |")
    w("")
    w(f"Replayed total, corrected: **{total/1000:.2f} s/fold** over {len(corrected)} instances.")
    w("")
    w("## Every point, paired-repeat sweep (matmul instances) — authoritative")
    w("")
    w("Each ratio is the median of three independent (incumbent, arm) pairs; `spread` is the range")
    w("across those three. The first row of every instance is the A/A control, the same estimator")
    w("with the arm replaced by another incumbent run. It lands at 0.989-1.006 on all seventeen,")
    w("which is the noise floor every other row has to clear.")
    w("")
    w("| instance | shipped config | shipped out | ms/fold | knob | ratio | spread | bit-exact |")
    w("|---|---|---|---|---|---|---|---|")
    for iid, i in rn.items():
        ms = corrected.get(iid, (0, ""))[0]
        ob = recs[iid]["out_mem"]["buffer"]
        first = True
        for arm in i["arms"]:
            head = (f"| `{iid}` | {i['shipped_config']} | {ob} | {ms:.0f} " if first
                    else "|  |  |  |  ")
            first = False
            if "error" in arm:
                w(head + f"| `{arm['knob']}` | {arm['error'][:58]} |  |  |")
                continue
            be = ""
            if arm["raw_knob"] == "outbuf=L1" and iid in bx:
                be = "**yes**" if bx[iid].get("bit_exact") else "no"
            w(head + f"| `{arm['knob']}` | {arm['ratio']:.3f}x | {arm['ratio_spread']:.3f} | {be} |")
    w("")
    w("## Every point, first sweep (non-matmul instances, unaffected by the straw man)")
    w("")
    w("| instance | op | ms/fold | knob | ratio | max_abs vs incumbent |")
    w("|---|---|---|---|---|---|")
    for iid, i in oldi.items():
        if recs[iid]["kind"] == "matmul":
            continue
        ms = corrected.get(iid, (0, ""))[0]
        first = True
        for arm in i.get("arms", []):
            head = f"| `{iid}` | {recs[iid]['kind']} | {ms:.0f} " if first else "|  |  |  "
            first = False
            if "error" in arm:
                w(head + f"| `{arm['knob']}` | {arm['error'][:60]} |  |")
            else:
                w(head + f"| `{arm['knob']}` | {arm['ratio']:.3f}x | {arm.get('max_abs','')} |")
        if not first:
            w(f"|  |  |  | **A/A floor** | **{i.get('aa_floor')}** |  |")
    pathlib.Path(a.out).write_text("\n".join(L) + "\n")
    print("wrote", a.out, f"({total/1000:.2f} s/fold corrected total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
