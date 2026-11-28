import json, sys, glob, os
fl = json.load(open(sys.argv[1]))
def recs(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if l.startswith("{") and "master_digest" in l:
                r = json.loads(l); out[r["batch"][5:7]] = r; break
    return out
arms = {a: recs(p) for a, p in zip(sys.argv[2::2], sys.argv[3::2])}
print(type(fl).__name__, list(fl)[:5] if isinstance(fl, dict) else fl[:1])
for term in ("bb_ca", "sequence_prediction", "insertion_prediction"):
    print(term)
    for b in sorted(next(iter(arms.values()))):
        row = [b]
        for a, R in arms.items():
            r = R.get(b)
            if not r: row.append(f"{a}:-"); continue
            d = abs(r["terms"][term] - r["upstream_terms"][term])
            row.append(f"{a}:{d:.2e}")
        print("  ", " ".join(row))
