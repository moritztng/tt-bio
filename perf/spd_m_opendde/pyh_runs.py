"""Contiguous MainThread runs (>= MIN s) whose leaf is NOT a ttnn op call: host-only stretches in which the device can starve."""
import json, sys, collections
p = json.load(open(sys.argv[1])); warm = float(sys.argv[2]); MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
fr = p["shared"]["frames"]; pr = next(x for x in p["profiles"] if "MainThread" in x["name"])
S, W = pr["samples"], pr["weights"]; tot = sum(W); acc = 0
for a, w in enumerate(W):
    acc += w
    if acc >= tot - warm - 1.0: break
def nm(i): f = fr[i]; return f'{f["name"]} {f.get("file","").split("/")[-1]}:{f.get("line","")}'
def devop(st):
    if not st: return False
    l = fr[st[-1]]
    return l["name"] == "__call__" and "decorators.py" in l.get("file", "")
t = 0.0; run = []; out = []
for st, w in zip(S[a:], W[a:]):
    if devop(st):
        if sum(x[1] for x in run) >= MIN: out.append((t - sum(x[1] for x in run), run))
        run = []
    else:
        run.append((st, w))
    t += w
tot_host = 0
for t0, r in out:
    d = sum(w for _, w in r); tot_host += d
    c = collections.Counter()
    for st, w in r:
        n = [nm(i) for i in st]; tb = [x for x in n if "site-packages" not in x and "threading" not in x]
        c[" < ".join(reversed(n[-4:]))] += w
    print(f"t={t0:7.2f}s  host-only {d:5.2f}s")
    for s, w in c.most_common(4): print(f"     {w:5.2f}s {s}")
print("total host-only runs >= MIN:", round(tot_host, 2))
