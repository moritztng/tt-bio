#!/usr/bin/env python3
"""The deliverable: accuracy against step count against measured seconds, with the knee marked.

Three tables, one file. Accuracy comes from the panel (10 targets x 13 settings, one WH chip
each, `sweep.py`); seconds come from the paired interleaved A/B (`paired_time.py`, one process,
one card, arms in turn). They are separate runs because they answer separate questions: the panel
needs 10 chips and does not care what a fold cost, the A/B needs one chip and cares about nothing
else.

Every accuracy cell is scored against that target's OWN seed floor -- the same fold re-run at the
SAME production setting with a different seed. Boltz-2's sampler is stochastic by construction, so
a cell that sits inside its target's floor has not been shown to be worse than production; a cell
scored against zero would report sampler chaos as a quality loss.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

STEPS = [(200, 3), (150, 3), (100, 3), (75, 3), (50, 3), (25, 3)]
RECYC = [(200, 2), (200, 1), (200, 0)]
CORNER = [(100, 2), (50, 2), (100, 1), (50, 1)]
ORDER = STEPS + [(200, 2), (100, 2), (50, 2), (200, 1), (100, 1), (50, 1), (200, 0)]


def table(d, metric, better_low):
    out = [f"{'target':>14} {'FLOOR':>7} {'kind':>16}"
           + "".join(f"{s}/{r:<1}".rjust(9) for s, r in ORDER)]
    misses = {k: 0 for k in ORDER}
    for t in d["targets"]:
        cells = {(r["steps"], r["recycles"]): r[metric] for r in t["rows"]
                 if r["seed"] == 0 and r["role"] != "floor"}
        floor = t["floor"]["rmsd_allatom_max_A" if metric == "rmsd_allatom_A"
                           else "lddt_ca_min"]
        line = ""
        for k in ORDER:
            v = cells.get(k)
            if v is None:
                line += "        -"
                continue
            bad = (v > floor) if better_low else (v < floor)
            if bad and k != (200, 3):
                misses[k] += 1
            line += f"{v:>8.2f}{'!' if bad else ' '}"
        out.append(f"{t['target']:>14} {floor:>7.2f} {str(t['kind']):>16}{line}")
    out.append(f"{'misses /10':>14} {'':>7} {'':>16}"
               + "".join(f"{'-' if k == (200,3) else misses[k]:>8} " for k in ORDER))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", type=Path, required=True)
    ap.add_argument("--paired", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.frontier.read_text())
    p = json.loads(a.paired.read_text())

    by = {}
    for r in p["runs"]:
        by.setdefault((r["steps"], r["recycles"]), []).append(r["fold_s"])
    inc = (p["env"]["arms"][0][0], p["env"]["arms"][0][1])
    inc_s = by[inc]
    aa = 100 * (max(inc_s) - min(inc_s)) / st.median(inc_s)

    L = []
    L.append("# Boltz-2: what 200 sampling steps and 3 recycles are worth\n")
    L.append(f"Accuracy: panel of {len(d['targets'])} targets x {len(ORDER)} settings, "
             f"one Wormhole chip each (`sweep.py`, `results/*.json`). ARCH: WH.")
    L.append(f"Seconds: paired interleaved A/B, {len(inc_s)} reps x {len(by)} arms in ONE process on "
             f"card {p['env']['card']} of {p['env']['host']}, {p['env']['target']}, ttnn "
             f"{p['env']['ttnn']}, grid {p['env'].get('grid')}. ARCH: WH.")
    L.append(f"A/A floor on the incumbent arm: {aa:.2f} % of its own median. A ratio inside that "
             "has not been measured.\n")

    L.append("## Accuracy: CA lDDT against this target's own (200,3,seed 0) fold\n")
    L.append("```\n" + table(d, "lddt_ca", better_low=False) + "\n```")
    L.append("`!` = outside this target's own seed floor: worse than re-running production "
             "unchanged with a different seed.\n")

    L.append("## Accuracy: all-atom RMSD (Angstrom) against the same reference\n")
    L.append("```\n" + table(d, "rmsd_allatom_A", better_low=True) + "\n```\n")

    L.append("## Seconds: measured, paired, same chip\n")
    L.append("```")
    L.append(f"{'arm':>9} {'n':>3} {'median_s':>9} {'ratio':>7} {'removed_s':>10}   reps")
    for k in ORDER:
        if k not in by:
            continue
        v = by[k]
        L.append(f"{f'{k[0]}/{k[1]}':>9} {len(v):>3} {st.median(v):>9.3f} "
                 f"{st.median(inc_s)/st.median(v):>7.4f} {st.median(inc_s)-st.median(v):>10.3f}   "
                 + " ".join(f"{x:.2f}" for x in v))
    L.append("```\n")

    L.append("## The frontier\n")
    L.append("```")
    L.append(f"{'arm':>9} {'median_s':>9} {'ratio':>7} {'misses/10':>10} {'lDDT median':>12} "
             f"{'worst RMSD_AA':>14}")
    mis_l = {}
    for t in d["targets"]:
        cells = {(r["steps"], r["recycles"]): r for r in t["rows"]
                 if r["seed"] == 0 and r["role"] != "floor"}
        for k, r in cells.items():
            if k == (200, 3):
                continue
            mis_l[k] = mis_l.get(k, 0) + (r["lddt_ca"] < t["floor"]["lddt_ca_min"])
    for k in ORDER:
        if k not in by:
            continue
        rows = [r for t in d["targets"] for r in t["rows"]
                if (r["steps"], r["recycles"]) == k and r["seed"] == 0 and r["role"] != "floor"]
        med = st.median(r["lddt_ca"] for r in rows)
        worst = max(r["rmsd_allatom_A"] for r in rows)
        L.append(f"{f'{k[0]}/{k[1]}':>9} {st.median(by[k]):>9.3f} "
                 f"{st.median(inc_s)/st.median(by[k]):>7.4f} "
                 f"{('-' if k == (200,3) else mis_l.get(k,0)):>10} {med:>12.2f} {worst:>14.3f}")
    L.append("```\n")
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
