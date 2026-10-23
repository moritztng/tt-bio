"""Fold elt_prof.py output at 512 into the census schema of state/rfd4/census/e2e-512-v5.json.
Recycled pass = the profiled pass that carries the recycle op; a thinking_medium design is 60 first + 120 recycled
passes. GFLOP per family comes from v5 (same shapes, the census is arithmetic).
  python3 perf/rfd4/land9/census_v6.py <census dir> <v5.json> <out.json> <sha> <card>"""
import json, sys
d, v5p, out, sha, card = sys.argv[1:]
t, run, v5 = json.load(open(f"{d}/table.json")), json.load(open(f"{d}/run.json")), json.load(open(v5p))
p = [x for x in t["passes"] if "recycle" in x["families"] or "tri_mult_in" in x["families"]]
rec = next(x for x in p if "recycle" in x["families"]); first = next(x for x in p if x is not rec)
nf, nr = v5["first_passes"], v5["recycled_passes"]
dev = (nf * first["device_ms"] + nr * rec["device_ms"]) / 1e3
fams = {}
for f in sorted(set(first["families"]) | set(rec["families"])):
    a, b = first["families"].get(f, {}), rec["families"].get(f, {})
    w = lambda k: (nf * a.get(k, 0) + nr * b.get(k, 0)) / (nf + nr)
    ms, gf = w("ms"), v5["families"].get(f, {}).get("gflop_per_pass", 0.0)
    fams[f] = {"ms_per_pass": round(ms, 4), "ops_per_pass": round(w("ops"), 1), "elementwise_ms_per_pass": round(w("elt_ms"), 4),
               "elementwise_ops_per_pass": round(w("elt_ops"), 1), "gflop_per_pass": gf,
               "tflops": round(gf / ms, 2) if ms else 0.0, "device_s_per_design": round(ms * (nf + nr) / 1e3, 3),
               "share_of_device": round(ms * (nf + nr) / 1e3 / dev, 4)}
fams = dict(sorted(fams.items(), key=lambda kv: -kv[1]["ms_per_pass"]))
o = {k: v5[k] for k in ("size", "recipe", "trunk_passes", "first_passes", "recycled_passes", "batch", "dims", "instrument")}
gf_pass = sum(x["gflop_per_pass"] for x in fams.values())
o.update(aiclk=run["aiclk"], host_load=round(run["load"], 1), card=f"qb2 physical {card}",
         per_pass={"first": {"device_ms": round(first["device_ms"], 3), "ops": first["ops"]},
                   "recycled": {"device_ms": round(rec["device_ms"], 3), "ops": rec["ops"], "elementwise_ms": round(rec["elementwise_ms"], 3),
                                "elementwise_ops": rec["elementwise_ops"]}},
         device_s_per_design=round(dev, 3), tflop_per_design=v5["tflop_per_design"], families=fams, sha=sha,
         note="qb2 p300c development number, never ratio-grade; row rfd4-land9", v5_device_s_per_design=v5["device_s_per_design"])
json.dump(o, open(out, "w"), indent=1); print(dev, {k: v["device_s_per_design"] for k, v in list(fams.items())[:5]})
