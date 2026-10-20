#!/usr/bin/env python3
"""Every device program of a devprof.py record, named: one row per family (ttnn verb x issuing tt_bio function).

    python perf/bc4_devgap/names.py _devgap/u1/devprof.json.gz [--json out.json] [--min 0.01]

device s is kernel time on the card's clock, per kept iteration (mean). bytes are the tensor operands and results
of the issuing calls (logical shapes, bfp8 at 1.0625 B), split evenly over the programs a call made, so a replayed
trace program carries its capture call's bytes. roof = max(bytes / 391 GB/s (p300c DRAM, bets/k128), FLOPs /
45 TFLOP/s): the op-local floor CEILING section 1 uses. An operand that lives in L1 makes the roof conservative.
"""
import argparse
import bisect
import collections
import gzip
import json

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--json")
ap.add_argument("--min", type=float, default=0.01)
ap.add_argument("--bw", type=float, default=391e9)
ap.add_argument("--rate", type=float, default=45e12)
a = ap.parse_args()
doc = json.load(gzip.open(a.path, "rt"))
C = {c: i for i, c in enumerate(doc["program_cols"])}
spans = sorted(tuple(s) for s in doc["spans"])
starts = [s[0] for s in spans]


def span_of(rid):
    i = bisect.bisect_right(starts, rid >> 10) - 1
    return i if i >= 0 and spans[i][0] <= rid >> 10 < spans[i][1] else None


progs_of = collections.defaultdict(set)     # span -> distinct runtime ids seen
for p in doc["programs"]:
    i = span_of(p[C["runtime_id"]])
    if i is not None:
        progs_of[i].add(p[C["runtime_id"]])

its = sorted({p[C["iteration"]] for p in doc["programs"]})
K = len(its)
fam = collections.defaultdict(lambda: dict(programs=0, k_ns=0, bytes=0.0, flops=0.0, sites=collections.Counter(),
                                           site_ns=collections.Counter()))
tot_n = tot_ns = 0
for p in doc["programs"]:
    i = span_of(p[C["runtime_id"]])
    if i is None:
        key, site, nb, fl = ("unmapped", "?"), "?", 0, 0
    else:
        s = spans[i]
        file, line, func = (s[3].split(":") + ["", ""])[:3] if s[3] != "?" else ("?", "", "?")
        key, site = (s[2], f"{file}:{func}"), f"{file}:{line}"
        share = 1 / max(1, len(progs_of[i]))
        nb, fl = (s[5] if len(s) > 5 else 0) * share, (s[6] if len(s) > 6 else 0) * share
    f = fam[key]
    k = p[C["k_ns"]] or 0
    f["programs"] += 1
    f["k_ns"] += k
    f["bytes"] += nb
    f["flops"] += fl
    f["sites"][site] += 1
    f["site_ns"][site] += k
    tot_n += 1
    tot_ns += k

rows = []
for (verb, where), f in fam.items():
    dev = f["k_ns"] / 1e9 / K
    by, fl = f["bytes"] / K, f["flops"] / K
    roof = max(by / a.bw, fl / a.rate)
    rows.append(dict(verb=verb, where=where, programs=f["programs"] / K, device_s=dev, p150a_s=0.80 * dev,
                     bytes=by, flops=fl, roof_s=roof, above_roof_s=dev - roof,
                     sites=[(s, f["site_ns"][s] / 1e9 / K) for s, _ in f["site_ns"].most_common(3)]))
rows.sort(key=lambda r: -r["device_s"])
um = fam.get(("unmapped", "?"), dict(programs=0, k_ns=0))
print(f"{doc['sha'][:9]} chip {doc['chip']} iterations {its}: {tot_n / K:.0f} programs/it, {tot_ns / 1e9 / K:.3f} device s/it; "
      f"unmapped {um['programs'] / K:.0f} programs ({um['programs'] / max(tot_n, 1):.2%}), {um['k_ns'] / 1e9 / K:.4f} s/it")
for m in doc["iterations"]:
    if m.get("kept"):
        print(f"  it {m['n']}: AICLK {m['aiclk'].get(str(doc['chip'])) or m['aiclk']} load {m['load_start']}->{m['load_end']}")
print(f"\n{'verb':34} {'issuing function':44} {'prog/it':>7} {'dev s':>7} {'p150a':>6} {'MB':>8} {'GFLOP':>7} "
      f"{'roof s':>7} {'above':>7}  top sites (device s)")
for r in rows:
    if r["device_s"] < a.min:
        continue
    print(f"{r['verb'][:34]:34} {r['where'][:44]:44} {r['programs']:7.0f} {r['device_s']:7.4f} {r['p150a_s']:6.3f} "
          f"{r['bytes'] / 1e6:8.1f} {r['flops'] / 1e9:7.1f} {r['roof_s']:7.4f} {r['above_roof_s']:7.4f}  "
          + ", ".join(f"{s} {t:.4f}" for s, t in r["sites"]))
small = [r for r in rows if r["device_s"] < a.min]
print(f"{'rest':34} {len(small):3d} families {'':40} {sum(r['programs'] for r in small):7.0f} "
      f"{sum(r['device_s'] for r in small):7.4f}")
verbs = collections.defaultdict(lambda: [0, 0.0, 0.0])
for r in rows:
    v = verbs[r["verb"]]
    v[0] += r["programs"]; v[1] += r["device_s"]; v[2] += r["above_roof_s"]
print(f"\n{'verb (all sites)':40} {'prog/it':>7} {'dev s':>7} {'above roof':>10}")
for v, (n, d, ab) in sorted(verbs.items(), key=lambda kv: -kv[1][1])[:40]:
    print(f"{v[:40]:40} {n:7.0f} {d:7.4f} {ab:10.4f}")
if a.json:
    json.dump(dict(source=a.path, sha=doc["sha"], chip=doc["chip"], iterations=its, programs_per_it=tot_n / K,
                   device_s_per_it=tot_ns / 1e9 / K, unmapped_programs_per_it=um["programs"] / K,
                   unmapped_s_per_it=um["k_ns"] / 1e9 / K, bw=a.bw, rate=a.rate, families=rows),
              open(a.json, "w"), indent=1)
