"""Census v2 from elt_prof's ops report: device kernel time and op count per family and per pass.

    elt_prof_table.py <ops_perf_results.csv> [--json out.json]

Rows between the `elt_prof` and `elt_end` signposts are charged to the innermost open `B|family`
signpost (`pass` alone means the trunk pass's own glue). Each `B|pass` starts a new pass. Elementwise
op codes are summed separately so the glue inside a family (casts, masks, residual adds) is visible.
"""
import collections
import json
import sys

ELT = ("Binary", "Unary", "Typecast", "Copy", "Clone")


def main():
    rows = list(csv_rows(sys.argv[1]))
    on, stack, passes = False, [], []
    for r in rows:
        if r.get("OP TYPE") == "signpost":
            code = r["OP CODE"]
            if code == "elt_prof":
                on = True
            elif code == "elt_end":
                on = False
            elif on and code.startswith("B|"):
                stack.append(code[2:])
                if code == "B|pass":
                    passes.append(collections.defaultdict(lambda: [0.0, 0, 0.0, 0]))
            elif on and code.startswith("E|"):
                stack.pop()
            continue
        if not on:
            continue
        p = passes[-1] if passes else collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
        if not passes:
            passes.append(p)
        fam = stack[-1] if stack else "outside_pass"
        fam = "pass_glue" if fam == "pass" else fam
        ms = float(r.get("DEVICE KERNEL DURATION [ns]") or 0) / 1e6
        cell = p[fam]
        cell[0] += ms
        cell[1] += 1
        if any(k in r["OP CODE"] for k in ELT):
            cell[2] += ms
            cell[3] += 1
    out = []
    for i, p in enumerate(passes):
        tot = sum(c[0] for c in p.values())
        ops = sum(c[1] for c in p.values())
        elt = sum(c[2] for c in p.values())
        eops = sum(c[3] for c in p.values())
        print(f"== pass {i}: device {tot:.3f} ms, {ops} ops; elementwise {elt:.3f} ms ({eops} ops)")
        for fam, c in sorted(p.items(), key=lambda kv: -kv[1][0]):
            print(f"   {fam:18s} {c[0]:8.3f} ms {c[1]:5d} ops   elementwise {c[2]:7.3f} ms {c[3]:4d} ops")
        out.append({"device_ms": tot, "ops": ops, "elementwise_ms": elt, "elementwise_ops": eops,
                    "families": {f: {"ms": c[0], "ops": c[1], "elt_ms": c[2], "elt_ops": c[3]} for f, c in p.items()}})
    by_code = collections.defaultdict(lambda: [0.0, 0])
    on = False
    for r in rows:
        if r.get("OP TYPE") == "signpost":
            on = {"elt_prof": True, "elt_end": False}.get(r["OP CODE"], on)
            continue
        if on:
            c = by_code[r["OP CODE"]]
            c[0] += float(r.get("DEVICE KERNEL DURATION [ns]") or 0) / 1e6
            c[1] += 1
    print("== by op code, whole measured call")
    for k, c in sorted(by_code.items(), key=lambda kv: -kv[1][0]):
        print(f"   {k[:48]:48s} {c[0]:8.3f} ms {c[1]:5d}")
    if "--json" in sys.argv:
        json.dump({"passes": out, "by_op_code": {k: {"ms": c[0], "ops": c[1]} for k, c in by_code.items()}},
                  open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


def csv_rows(path):
    import csv
    return csv.DictReader(open(path))


if __name__ == "__main__":
    main()
