"""Per-op table of census families: `python3 op_table.py <prof dir> [--call] [family ...]`.

Reads a census_v2 profile (ops.csv from elt_prof.py or call_prof.py) with the same signpost attribution as
census_v2.py and prints every op of the last (recycled) pass in the named families (all when none is named) with
op code, input/output shapes, dtypes, layout and device kernel time, so a family's ops can be named one by one.
`--call` covers the whole profiled window instead (call_prof.py: pair build, trunk entry, recycle and readback
included) and adds the device span, whose excess over the kernel sum is the gaps between kernels.
"""
import collections
import csv
import json
import sys


def shape(r, k):
    d = [r.get(f"{k}_{a}_PAD[LOGICAL]", "") for a in "WZYX"]
    if not any(d):
        return ""
    d = [x.split("[")[-1].rstrip("]") if "[" in x else x for x in d]
    dt = r.get(k + "_DATATYPE", "").replace("DataType::", "")
    return "x".join(d) + f" {dt} {r.get(k + '_LAYOUT', '').replace('Layout::', '')}"


def main():
    args = sys.argv[2:]
    whole = "--call" in args
    prof, want = sys.argv[1], {a for a in args if a != "--call"}
    on, stack, passes = False, [], [[]] if whole else []
    for r in csv.DictReader(open(f"{prof}/ops.csv")):
        if r["OP TYPE"] == "signpost":
            c = r["OP CODE"]
            on = {"elt_prof": True, "elt_end": False}.get(c, on)
            if on and c.startswith("B|"):
                stack.append(c[2:])
                if c == "B|pass" and not whole:
                    passes.append([])
            elif on and c.startswith("E|"):
                stack.pop()
            continue
        if not on or ("pass" not in stack and not whole):
            continue
        fam = stack[-1] if stack else "outside"
        fam = "pass_glue" if fam == "pass" else fam
        passes[-1].append((fam, r))
    rows = [(f, r) for f, r in passes[-1] if not want or f in want]
    agg = collections.defaultdict(lambda: [0, 0.0])
    for f, r in rows:
        ms = float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e6
        key = (f, r["OP CODE"], shape(r, "INPUT_0"), shape(r, "INPUT_1"), shape(r, "OUTPUT_0"), r["CORE COUNT"])
        agg[key][0] += 1
        agg[key][1] += ms
    out = []
    for (f, op, i0, i1, o0, cores), (cnt, ms) in sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[1][1])):
        out.append({"family": f, "op": op, "count": cnt, "ms": round(ms, 4), "ms_each": round(ms / cnt, 4),
                    "in0": i0, "in1": i1, "out0": o0, "cores": cores})
        print(f"{f:12s} {op:34s} x{cnt:<3d} {ms:8.3f} ms ({ms / cnt:6.3f} each) cores {cores:>3s}  "
              f"in0 {i0} | in1 {i1} | out {o0}")
    print(f"total {sum(v[1] for v in agg.values()):.3f} ms device kernel, {sum(v[0] for v in agg.values())} ops")
    if whole:
        fam = collections.defaultdict(float)
        for k, v in agg.items():
            fam[k[0]] += v[1]
        for f, ms in sorted(fam.items(), key=lambda kv: -kv[1]):
            print(f"  {f:16s} {ms:9.3f} ms")
        ns = [(int(r["DEVICE FW START CYCLE"]), int(r["DEVICE FW END CYCLE"])) for _, r in rows
              if r["DEVICE FW START CYCLE"] and r["DEVICE FW END CYCLE"]]
        if ns:
            print(f"device span {(max(e for _, e in ns) - min(s for s, _ in ns)) / 1350e3:.3f} ms "
                  "at 1350 MHz (first fw start to last fw end)")
    json.dump(out, open(f"{prof}/op_table{'_call' if whole else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
