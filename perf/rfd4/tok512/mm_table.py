"""Per-label device time of mm_sweep.py, and the best arm per (site, M) against the trunk's call.

    python3 perf/rfd4/tok512/mm_table.py <sweep dir>

us/call = DEVICE KERNEL DURATION summed inside `B|label` over the reps. The trunk's call is `bf16|HiFi4|linear`.
"""
import collections
import csv
import json
import sys


def main():
    d = sys.argv[1]
    sw = json.load(open(f"{d}/sweep.json"))
    reps = sw["reps"]
    us, label = collections.defaultdict(float), None
    for r in csv.DictReader(open(f"{d}/ops.csv")):
        if r["OP TYPE"] == "signpost":
            c = r["OP CODE"]
            label = c[2:] if c.startswith("B|") else (None if c.startswith("E|") else label)
            continue
        if label:
            us[label] += float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3 / reps
    err = {x["label"]: x.get("err") for x in sw["rows"]}
    groups = collections.defaultdict(list)
    for lab, t in us.items():
        site, m, wd, fid, impl = lab.split("|")
        groups[(site, int(m))].append((t, wd, fid, impl, err.get(lab)))
    out = []
    print(f"AICLK {sw['aiclk']}  load {sw['load']}")
    for (site, m), rows in sorted(groups.items()):
        base = next((r for r in rows if r[1:4] == ("bf16", "HiFi4", "linear")), None)
        print(f"== {site} M={m}  trunk call {base[0]:.1f} us err {base[4]}" if base else f"== {site} M={m}")
        best = {}
        for r in sorted(rows):
            k = (r[1], r[2], r[3] == "linear")
            if k not in best:
                best[k] = r
        for k, r in sorted(best.items(), key=lambda kv: kv[1][0]):
            print(f"   {r[1]:4s} {r[2]:5s} {r[3]:18s} {r[0]:8.1f} us  err {r[4] if r[4] is None else round(r[4], 5)}" + (
                f"  x{base[0] / r[0]:.2f}" if base else ""))
            out.append(dict(site=site, m=m, w=r[1], fid=r[2], impl=r[3], us=round(r[0], 2), err=r[4],
                            trunk_us=base and round(base[0], 2)))
    json.dump(out, open(f"{d}/table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
