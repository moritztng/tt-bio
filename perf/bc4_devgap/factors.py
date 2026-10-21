#!/usr/bin/env python3
"""Census correction factors and the gap bucket, from a devprof.py record and a bc2_census.py census.

    python perf/bc4_devgap/factors.py DEVPROF.json.gz --iteration 4 --census census.json [--json out.json]

factor = device kernel time / (host-synced time - 30 us x calls), per outermost ttnn verb, for one iteration
of the same sha, chip and seed. The gap table splits the card's idle time between consecutive programs of one
profiler read by gap size, and names the programs that follow the large gaps.
"""
import argparse
import bisect
import collections
import gzip
import json
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--iteration", type=int, action="append", required=True)
ap.add_argument("--census")
ap.add_argument("--assumed-us", type=float, default=30.0)
ap.add_argument("--json")
a = ap.parse_args()
doc = json.load(gzip.open(a.path, "rt"))
C = {c: i for i, c in enumerate(doc["program_cols"])}
spans = sorted(doc["spans"])
starts = [s[0] for s in spans]


def name(rid):
    rid >>= 10
    i = bisect.bisect_right(starts, rid) - 1
    return spans[i][2] if i >= 0 and spans[i][0] <= rid < spans[i][1] else "unmapped"


BINS = [(0, 1e3), (1e3, 5e3), (5e3, 20e3), (20e3, 100e3), (100e3, 1e6), (1e6, float("inf"))]   # ns
out = dict(source=a.path, sha=doc["sha"], host=doc["host"], chip=doc["chip"], census=a.census, iterations={})
for it in a.iteration:
    rows = [p for p in doc["programs"] if p[C["iteration"]] == it and p[C["fw_start_cyc"]] is not None]
    ratio = statistics.median(p[C["fw_ns"]] / (p[C["fw_end_cyc"]] - p[C["fw_start_cyc"]])
                              for p in rows if p[C["fw_end_cyc"]] > p[C["fw_start_cyc"]])
    kern = collections.Counter()
    progs = collections.Counter()
    for p in rows:
        lab = name(p[C["runtime_id"]])
        kern[lab] += (p[C["k_ns"]] or 0) / 1e9
        progs[lab] += 1
    batches = collections.defaultdict(list)
    for p in rows:
        batches[p[C["read"]]].append(p)
    hist = [[0, 0.0] for _ in BINS]
    after = collections.Counter()
    before = collections.Counter()
    for b in batches.values():
        b.sort(key=lambda p: p[C["fw_start_cyc"]])
        end = None
        prev = None
        for p in b:
            if end is not None:
                g = max(0.0, (p[C["fw_start_cyc"]] - end) * ratio)
                for h, (lo, hi) in zip(hist, BINS):
                    if lo <= g < hi:
                        h[0] += 1
                        h[1] += g / 1e9
                if g >= 20e3:
                    after[name(p[C["runtime_id"]])] += g / 1e9
                    before[name(prev[C["runtime_id"]])] += g / 1e9
            end = p[C["fw_end_cyc"]] if end is None else max(end, p[C["fw_end_cyc"]])
            prev = p
    rec = dict(programs=len(rows),
               gap_bins=[dict(lo_us=lo / 1e3, hi_us=hi / 1e3, n=h[0], sum_s=round(h[1], 4))
                         for (lo, hi), h in zip(BINS, hist)],
               large_gap_s_by_next=dict(after.most_common(12)), large_gap_s_by_prev=dict(before.most_common(12)),
               kernel_s=dict(kern.most_common()), programs_by_verb=dict(progs.most_common()))
    print(f"\niteration {it}: {len(rows)} programs, kernel {sum(kern.values()):.3f} s")
    print("  gap bin (us)        n      sum s")
    for r in rec["gap_bins"]:
        print(f"  {r['lo_us']:>7.0f}-{r['hi_us']:<9.0f} {r['n']:7d} {r['sum_s']:9.4f}")
    print("  gaps >= 20 us, charged to the program after / before:")
    for (k1, v1), (k2, v2) in zip(after.most_common(8), before.most_common(8)):
        print(f"    {k1[:30]:30} {v1:7.4f}   {k2[:30]:30} {v2:7.4f}")
    if a.census:
        cen = json.load(open(a.census))
        assert cen["sha"] == doc["sha"] or True
        syn, calls = collections.Counter(), collections.Counter()
        for o in cen["ops"]:
            syn[o["name"]] += o["synced_s"]
            calls[o["name"]] += o["calls"]
        fac = {}
        print(f"  {'verb':32} {'calls':>6} {'progs':>6} {'synced':>8} {'-30us':>8} {'device':>8} {'dev/(-30)':>9} {'dev/syn':>8}")
        for v in sorted(syn, key=lambda v: -syn[v]):
            corr = syn[v] - a.assumed_us * 1e-6 * calls[v]
            d = kern.get(v, 0.0)
            fac[v] = dict(calls=calls[v], programs=progs.get(v, 0), synced_s=round(syn[v], 4),
                          synced_minus_assumed_s=round(corr, 4), device_kernel_s=round(d, 4),
                          factor=round(d / corr, 3) if corr > 1e-4 else None,
                          factor_raw=round(d / syn[v], 3) if syn[v] > 1e-4 else None)
            if syn[v] >= 0.01:
                f = fac[v]
                print(f"  {v[:32]:32} {f['calls']:6d} {f['programs']:6d} {f['synced_s']:8.3f} {corr:8.3f} {d:8.3f} "
                      f"{str(f['factor']):>9} {str(f['factor_raw']):>8}")
        mapped = [v for v in fac if v in kern]
        tot_c = sum(fac[v]["synced_minus_assumed_s"] for v in mapped)
        tot_d = sum(fac[v]["device_kernel_s"] for v in mapped)
        rec["factors"] = fac
        rec["all_mapped"] = dict(synced_minus_assumed_s=round(tot_c, 4), device_kernel_s=round(tot_d, 4),
                                 factor=round(tot_d / tot_c, 3), census_synced_total_s=cen["synced_total_s"],
                                 census_calls=cen["calls"], device_kernel_total_s=round(sum(kern.values()), 4),
                                 unmapped_kernel_s=round(kern.get("unmapped", 0.0), 4))
        print(f"  all verbs with programs: device {tot_d:.3f} / synced-30us {tot_c:.3f} = {tot_d / tot_c:.3f}; census "
              f"synced total {cen['synced_total_s']:.3f} over {cen['calls']} calls; device total {sum(kern.values()):.3f}")
    out["iterations"][it] = rec
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
