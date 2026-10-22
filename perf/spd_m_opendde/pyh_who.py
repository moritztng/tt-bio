import json, sys, collections
p = json.load(open(sys.argv[1])); warm = float(sys.argv[2]); keys = sys.argv[3:]
fr = p["shared"]["frames"]; pr = next(x for x in p["profiles"] if "MainThread" in x["name"])
S, W = pr["samples"], pr["weights"]; tot = sum(W); acc = 0
for a, w in enumerate(W):
    acc += w
    if acc >= tot - warm - 1.0: break
def nm(i): f = fr[i]; return f'{f["name"]} {f.get("file","").split("/")[-1]}:{f.get("line","")}'
for k in keys:
    c = collections.Counter()
    for st, w in zip(S[a:], W[a:]):
        n = [nm(i) for i in st]
        if any(k in x for x in n):
            j = max(i for i, x in enumerate(n) if k in x)
            c[" < ".join(reversed([x for x in n[max(0,j-5):j+2] ]))] += w
    print(f"== {k}  {sum(c.values()):.2f}s")
    for s, w in c.most_common(8): print(f"  {w:6.2f}s {s}")
