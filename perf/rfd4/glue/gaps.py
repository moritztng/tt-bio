"""Device gaps of a call_prof window: idle time between consecutive kernels, charged to the family of the op after
the gap, plus the largest single gaps with the op on each side. `python3 gaps.py <prof dir>`."""
import collections
import csv
import json
import sys


def main():
    prof = sys.argv[1]
    run = json.load(open(f"{prof}/run.json"))
    mhz = next(iter(run["aiclk"].values()))["median"]
    on, stack, ops = False, [], []
    for r in csv.DictReader(open(f"{prof}/ops.csv")):
        if r["OP TYPE"] == "signpost":
            c = r["OP CODE"]
            on = {"elt_prof": True, "elt_end": False}.get(c, on)
            if on and c.startswith("B|"):
                stack.append(c[2:])
            elif on and c.startswith("E|"):
                stack.pop()
            continue
        if on and r["DEVICE FW START CYCLE"]:
            fam = stack[-1] if stack else "outside"
            ops.append((int(r["DEVICE FW START CYCLE"]), int(r["DEVICE FW END CYCLE"]),
                        "pass_glue" if fam == "pass" else fam, r["OP CODE"]))
    ops.sort()
    by, big = collections.defaultdict(float), []
    for a, b in zip(ops, ops[1:]):
        g = max(0, b[0] - a[1]) / (mhz * 1e3)
        by[b[2]] += g
        big.append((g, a[2], a[3], b[2], b[3]))
    span = (ops[-1][1] - ops[0][0]) / (mhz * 1e3)
    busy = sum(b - a for a, b, *_ in ops) / (mhz * 1e3)
    print(f"AICLK {mhz} MHz: span {span:.3f} ms, fw busy {busy:.3f} ms, gaps {sum(by.values()):.3f} ms, {len(ops)} ops")
    for f, g in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  gap before {f:16s} {g:8.3f} ms")
    for g, fa, oa, fb, ob in sorted(big, reverse=True)[:12]:
        print(f"  {g:8.3f} ms  {fa}/{oa} -> {fb}/{ob}")


if __name__ == "__main__":
    main()
