"""Per-op device table inside the tri_mult_in / tri_mult_out brackets of an elt_prof Tracy CSV.

    python3 optable.py <ops.csv> [--json out.json]

Charges each op to the innermost open B|/E| family signpost (as elt_prof_table.py does), then groups the ops of
every call of a family by their position in the call, and prints the median DEVICE KERNEL DURATION per position
with the op code, core count, fidelity and the operand shapes / dtypes / memory, so a call reads as one row each.
"""
import csv
import json
import statistics
import sys
from collections import defaultdict


def shape(r, i):
    d = [r.get(f"INPUT_{i}_{a}_PAD[LOGICAL]", "") for a in "WZYX"]
    if not any(d):
        return ""
    return "x".join(x.split("[")[0] for x in d) + f" {r.get(f'INPUT_{i}_DATATYPE','')[:6]} {r.get(f'INPUT_{i}_MEMORY','').replace('DEV_0_','')[:12]}"


def oshape(r):
    d = [r.get(f"OUTPUT_0_{a}_PAD[LOGICAL]", "") for a in "WZYX"]
    return "x".join(x.split("[")[0] for x in d) + f" {r.get('OUTPUT_0_DATATYPE','')[:6]} {r.get('OUTPUT_0_MEMORY','').replace('DEV_0_','')[:12]}"


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    stack, calls = [], defaultdict(list)
    for r in rows:
        if r.get("OP TYPE") == "signpost":
            c = r["OP CODE"]
            if c.startswith("B|"):
                stack.append(c[2:])
                if c[2:].startswith("tri_mult"):
                    calls[c[2:]].append([])
            elif c.startswith("E|") and stack:
                stack.pop()
            continue
        fam = next((f for f in reversed(stack) if f.startswith("tri_mult")), None)
        if fam and stack and stack[-1].startswith("tri_mult"):
            calls[fam][-1].append(r)
    out = {}
    for fam, cs in sorted(calls.items()):
        n = len(cs)
        lens = {len(c) for c in cs}
        print(f"== {fam}: {n} calls, ops per call {sorted(lens)}")
        L = max(lens)
        tab = []
        tot = 0.0
        for i in range(L):
            rs = [c[i] for c in cs if len(c) == L]
            ms = statistics.median(float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e6 for r in rs)
            r0 = rs[-1]
            tot += ms
            row = {"pos": i, "op": r0["OP CODE"], "ms": round(ms, 4), "cores": r0["CORE COUNT"],
                   "fidelity": r0["MATH FIDELITY"], "in0": shape(r0, 0), "in1": shape(r0, 1), "in2": shape(r0, 2),
                   "out": oshape(r0)}
            tab.append(row)
            print(f"  {i:2d} {row['op'][:34]:34s} {ms:7.3f} ms cores {row['cores']:>4s} {row['fidelity'][:6]:6s} "
                  f"| {row['in0']} | {row['in1']} | {row['in2']} -> {row['out']}")
        print(f"  sum {tot:.3f} ms/call")
        out[fam] = {"calls": n, "ops": tab, "ms_per_call": round(tot, 4)}
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
