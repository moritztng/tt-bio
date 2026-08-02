import json, glob, csv, collections, statistics

labels = {}
for p in glob.glob("/home/ttuser/abag_xm/tier_a/labels/opendde_abag_*.json"):
    d = json.load(open(p))
    dq = []
    for s in d["samples"]:
        v = s.get("dockq")
        if isinstance(v, dict) and v.get("dockq") is not None:
            dq.append(float(v["dockq"]))
    labels[d["target"]] = dq

sub = collections.defaultdict(set)
with open("/home/ttuser/abag_xm/OpenDDE/benchmarks/2026ARK_AB/common_interfaces.csv") as f:
    for r in csv.DictReader(f):
        sub[r["pdb_id"]].add(r["subset"])

def stratum(t):
    j = ";".join(sorted(sub.get(t, set())))
    if "scFv" in j:
        return "scFv"
    if "antibody_HL" in j:
        return "paired-Fv"
    if "antibody_H-protein" in j:
        return "VHH"
    return "other"

rows = []
for t, dq in sorted(labels.items()):
    if not dq:
        continue
    s50 = sum(1 for x in dq if x >= 0.23)
    rows.append(dict(t=t, stratum=stratum(t), n=len(dq), s50=s50,
                     best=max(dq), sd=statistics.pstdev(dq) if len(dq) > 1 else 0.0))

fails = [r for r in rows if r["best"] < 0.23]
succ = [r for r in rows if r["best"] >= 0.23]
print(f"targets {len(rows)}  fail {len(fails)}  succeed {len(succ)}  oracle@50 = {100.0*len(succ)/len(rows):.1f}%")
print("fail strata:", dict(collections.Counter(r["stratum"] for r in fails)))
print("succ strata:", dict(collections.Counter(r["stratum"] for r in succ)))

def fbuck(r):
    return "near" if r["best"] >= 0.18 else ("mid" if r["best"] >= 0.10 else "far")

def sbuck(r):
    return "fragile" if r["s50"] <= 5 else ("mid" if r["s50"] <= 25 else "robust")

print("\nFAIL buckets (best desc):")
for b in ["near", "mid", "far"]:
    bb = sorted([r for r in fails if fbuck(r) == b], key=lambda r: (-r["best"], r["t"]))
    c = collections.Counter(r["stratum"] for r in bb)
    print(f" {b} ({len(bb)}):  strata {dict(c)}")
    for r in bb:
        print(f"   {r['t']} {r['stratum']:9s} best={r['best']:.4f} s50={r['s50']} sd={r['sd']:.4f}")

print("\nSUCCEED buckets (s50 asc):")
for b in ["fragile", "mid", "robust"]:
    bb = sorted([r for r in succ if sbuck(r) == b], key=lambda r: (r["s50"], r["t"]))
    c = collections.Counter(r["stratum"] for r in bb)
    print(f" {b} ({len(bb)}):  strata {dict(c)}")
    for r in bb[:12]:
        print(f"   {r['t']} {r['stratum']:9s} best={r['best']:.4f} s50={r['s50']} sd={r['sd']:.4f}")
