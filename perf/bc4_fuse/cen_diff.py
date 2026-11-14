import json, sys, collections
a, b = (json.load(open(f"{sys.argv[1]}/{x}/devprof.json"))["census"] for x in ("off_cen", "l1_cen"))
def agg(c):
    d = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r in c["ops"]:
        k = (r["phase"], r["name"])
        d[k][0] += r["calls"]; d[k][1] += r["synced_s"]; d[k][2] += r["bytes"]
    return d
A, B = agg(a), agg(b)
print("iteration_s", a["iteration_s"], b["iteration_s"], "calls", a["calls"], b["calls"])
for ph in sorted({k[0] for k in A} | {k[0] for k in B}):
    sa = sum(v[1] for k, v in A.items() if k[0] == ph); sb = sum(v[1] for k, v in B.items() if k[0] == ph)
    ca = sum(v[0] for k, v in A.items() if k[0] == ph); cb = sum(v[0] for k, v in B.items() if k[0] == ph)
    ga = sum(v[2] for k, v in A.items() if k[0] == ph) / 1e9; gb = sum(v[2] for k, v in B.items() if k[0] == ph) / 1e9
    print(f"{ph:22} synced {sa:7.3f} -> {sb:7.3f}  calls {ca:6d} -> {cb:6d}  GB {ga:6.1f} -> {gb:6.1f}")
rows = []
for k in set(A) | set(B):
    va, vb = A.get(k, [0, 0, 0]), B.get(k, [0, 0, 0])
    if va[0] != vb[0] or abs(va[2] - vb[2]) > 1e8:
        rows.append((vb[1] - va[1], k, va, vb))
for d, k, va, vb in sorted(rows):
    print(f"{k[0]:20} {k[1]:28} calls {va[0]:5d}->{vb[0]:5d} synced {va[1]:6.3f}->{vb[1]:6.3f} ({d:+.3f}) "
          f"GB {va[2]/1e9:6.1f}->{vb[2]/1e9:6.1f} us/call {1e6*va[1]/max(va[0],1):6.0f}->{1e6*vb[1]/max(vb[0],1):6.0f}")
