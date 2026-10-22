#!/usr/bin/env python3
"""bc4-bet-mm2: price each lever from bench.py's out.json, per site, in device s per gradient iteration.

    python3 -I analyze.py out.json
"""
import json
import sys

d = json.load(open(sys.argv[1]))
T = "dev_min"          # traced burst, fastest replay: device time without host enqueue or sync jitter
LEVERS = {
    "hifi2": ("hifi2",),
    "bfp8": ("bfp8",),
    "hifi2_bfp8": ("hifi2_bfp8", "hifi2", "bfp8"),
    "mmm_bf16": ("best_mmm",),
    "all": ("hifi2_bfp8", "hifi2", "bfp8", "best_mmm", "best_mmm_hifi2", "best_mmm_bfp8", "best_mmm_hifi2_bfp8"),
}
site = {}
top = []
for r in d["rows"]:
    arms = {a["arm"]: a for a in r["arms"] if T in a}
    base = arms.get("base")
    if base is None:
        print("NO BASE", r["site"], r["x"], r["y"], [a.get("error") for a in r["arms"]][:1])
        continue
    mm = [a for k, a in arms.items() if k.startswith("mmm")]
    if mm:
        arms["best_mmm"] = min(mm, key=lambda a: a[T])
    n = r["calls_per_it"]
    s = site.setdefault(r["site"], dict(calls=0, base=0.0, sync=0.0, **{k: 0.0 for k in LEVERS}))
    s["calls"] += n
    s["base"] += n * base[T] / 1e3
    s["sync"] += n * base["sync_ms"] / 1e3
    best = {}
    for lev, names in LEVERS.items():
        cands = [arms[k][T] for k in names if k in arms] + [base[T]]
        t = min(cands)
        s[lev] += n * (base[T] - t) / 1e3
        best[lev] = t
    top.append((n * base[T], r["site"], n, r["mkn"], base[T], best, {k: (a[T], a.get("rel_f64")) for k, a in arms.items()
                                                                      if k in ("base", "hifi2", "bfp8", "hifi2_bfp8", "best_mmm",
                                                                               "best_mmm_bfp8")}))
for k, s in site.items():
    print(k, {kk: round(v, 4) if isinstance(v, float) else v for kk, v in s.items()})
print("saving s/it p300c device (all = best arm per row incl. bfp8 out):",
      round(sum(s["all"] for s in site.values()), 4), "p150a x0.80:", round(0.8 * sum(s["all"] for s in site.values()), 4))
top.sort(key=lambda t: -t[0])
for t in top[:20]:
    print(f"{t[0]:7.2f} ms/it {t[1]:9} n={t[2]:5.0f} bMKN={t[3]} base={t[4]:.4f} "
          + " ".join(f"{k}={v[0]:.4f}/{(v[1] or 0):.4f}" for k, v in t[6].items()))
print("aiclk", d.get("aiclk"), "load", d.get("loadavg"), "sha", d["sha"][:9], "dirty", d["dirty"])
