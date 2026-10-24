"""Per-call head and tail from call_prof's ops.csv: device kernel ms per signpost family, per trunk call.

    call_table.py <prof_dir>      prints the table, writes <prof_dir>/table.json

A call is one `B|launch` (plus the `B|collect` that reads it back). Its head is every op inside the launch but
outside `trunk`; its tail is `readback` (wherever it ran) and whatever `collect` enqueues. The per-design figure
multiplies the per-call median by calls a design (`--calls`, 60 at thinking_medium and lane width 1) and divides by
the designs a call carries (the lane width).
"""
import collections
import csv
import json
import statistics
import sys

CALLS_PER_DESIGN = 60


def main():
    d = sys.argv[1]
    run = json.load(open(f"{d}/run.json"))
    on, stack, calls, cur = False, [], [], None
    codes = collections.defaultdict(lambda: [0.0, 0])
    for r in csv.DictReader(open(f"{d}/ops.csv")):
        if r.get("OP TYPE") == "signpost":
            c = r["OP CODE"]
            on = {"elt_prof": True, "elt_end": False}.get(c, on)
            if on and c.startswith("B|"):
                stack.append(c[2:])
                if c == "B|launch":
                    cur = collections.defaultdict(float)
                    calls.append(cur)
            elif on and c.startswith("E|") and stack:
                stack.pop()
            continue
        if not on or cur is None:
            continue
        ms = float(r.get("DEVICE KERNEL DURATION [ns]") or 0) / 1e6
        fam = next((f for f in reversed(stack) if f in ("trunk", "readback", "upload", "masks", "pair_build")),
                   None) or (stack[-1] if stack else "between_calls")
        cur[fam] += ms
        if fam != "trunk":
            codes[(fam, r["OP CODE"])][0] += ms
            codes[(fam, r["OP CODE"])][1] += 1
    fams = sorted({f for c in calls for f in c})
    med = {f: statistics.median(c.get(f, 0.0) for c in calls) for f in fams}
    head_tail = sum(v for f, v in med.items() if f != "trunk")
    lane = run.get("lane", 1)
    per_design = head_tail * CALLS_PER_DESIGN / lane / 1e3
    print(f"{len(calls)} calls, n {run['n']} batch {run['batch']} lane {lane}, AICLK {run['aiclk']}")
    for f in fams:
        print(f"  {f:14s} median {med[f]:8.3f} ms a call")
    print(f"head+tail (all but trunk): {head_tail:.3f} ms a call = {per_design:.3f} s a design")
    print("ops outside the trunk (all calls):")
    for (f, c), (ms, k) in sorted(codes.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"  {f:12s} {c:40s} {ms:9.3f} ms {k:5d} ops")
    json.dump({"calls": len(calls), "median_ms_a_call": med, "head_tail_ms_a_call": head_tail,
               "head_tail_s_a_design": per_design, "run": run,
               "ops": [{"family": f, "op": c, "ms": ms, "n": k} for (f, c), (ms, k) in codes.items()]},
              open(f"{d}/table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
