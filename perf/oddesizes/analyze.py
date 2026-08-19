import json, math, sys
from pathlib import Path

D = Path("/home/ttuser/.coworker/wt/sizes-recheck-opendde/perf/oddesizes")
LEGS = {128: "ladder_128_256_512_qb1c2.json", 256: "ladder_128_256_512_qb1c2.json",
        512: "ladder_128_256_512_qb1c2.json", 768: "ladder_768_qb1c2.json",
        1024: "ladder_1024_qb1c2.json", 640: "offlattice_640_qb1c2.json"}
HIST = {128: 14.501, 256: 28.928, 512: 88.756, 768: 267.500, 1024: 705.498}  # 08-13, qb1 card 1, 0.68.0

runs = {}
meta = {}
for size, fn in LEGS.items():
    p = D / fn
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    meta[size] = {k: d.get(k) for k in ("ttnn", "host", "card", "grid")}
    for r in d["runs"]:
        runs.setdefault(size, []).append(r)

def on_arms(size):
    return [r for r in runs.get(size, []) if r.get("arm") == "on" and "error" not in r]

print("== acceptance ==")
for size in sorted(meta):
    m = meta[size]
    bad = []
    if m["ttnn"] != "0.68.0": bad.append("ttnn=%s" % m["ttnn"])
    if m["grid"] != [13, 10]: bad.append("grid=%s" % m["grid"])
    if m["card"] != "2": bad.append("card=%s" % m["card"])
    for r in on_arms(size):
        pm = r["persistent_mask"]
        if not pm["q_split"]: bad.append("q_split not on")
        if r["transpose_l1_headroom"] != 1.25: bad.append("headroom=%s" % r["transpose_l1_headroom"])
    errs = [r for r in runs.get(size, []) if "error" in r]
    if errs: bad.append("errors: %s" % [(e["arm"], e["error"][:80]) for e in errs])
    print("  %s: %s" % (size, "OK" if not bad else "FAIL " + "; ".join(bad)))

print("\n== rung table ==")
print("%5s %28s %9s %7s %9s %8s %18s" % ("size", "on arms (s)", "median", "A/A", "noqsplit", "dqsplit", "cif"))
table = {}
for size in sorted(runs):
    ons = on_arms(size)
    if not ons: continue
    times = sorted(r["fold_s"] for r in ons)
    med = times[len(times)//2] if len(times) % 2 else sum(times)/len(times)
    aa = max(times) - min(times)
    nq = [r for r in runs[size] if r.get("arm") == "noqsplit" and "error" not in r]
    nqs = "%.3f" % nq[0]["fold_s"] if nq else "-"
    dq = "%+.3f" % (nq[0]["fold_s"]-med) if nq else "-"
    cifs = {r.get("cif_sha256", "")[:16] for r in runs[size] if "error" not in r}
    table[size] = med
    print("%5s %28s %9.3f %7.3f %9s %8s %18s" % (size, str(["%.3f" % t for t in times]), med, aa, nqs, dq, str(cifs)))

print("\n== exponents (consecutive rungs, today vs 08-13) ==")
ladder = [s for s in (128, 256, 512, 768, 1024) if s in table]
for a, b in zip(ladder, ladder[1:]):
    e_now = math.log(table[b]/table[a]) / math.log(b/a)
    e_then = math.log(HIST[b]/HIST[a]) / math.log(b/a)
    print("  %4d->%-4d today N^%.3f   08-13 N^%.3f" % (a, b, e_now, e_then))

print("\n== vs 08-13 (same host; 08-13 was card 1, today card 2) ==")
for s in ladder:
    d = table[s] - HIST[s]
    print("  %5d: today %9.3f  08-13 %9.3f  delta %+9.3f (%+.2f%%)" % (s, table[s], HIST[s], d, 100*d/HIST[s]))

print("\n== census (first on arm per size) ==")
for size in sorted(runs):
    ons = on_arms(size)
    if not ons: continue
    r = ons[0]
    pm = r["persistent_mask"]; hq = r["head_major_qkv"]; gk = r["gated_kernel"]
    print("  %s: pm served=%s declined=%s qsplit=%s" % (size, pm["served"], pm["declined"], pm["q_split"]))
    if pm["rejects"]: print("        pm rejects: %s" % dict(list(pm["rejects"].items())[:6]))
    if pm["pm_over_l1"]: print("        pm_over_l1: %s" % pm["pm_over_l1"])
    print("        K1 served=%s declined=%s tail=%s/%s  E6 gated=%s" % (hq["served"], hq["declined"], hq["tail_served"], hq["tail_declined"], gk))
    print("        sdpa_q_chunk_over_l1: %s" % r["sdpa_q_chunk_over_l1"])
    dec = r.get("decisions", {})
    for k in sorted(dec):
        if "transition" in k or "transpose" in k or "pair_proj" in k:
            print("        dec %s: %s" % (k, dec[k]))
