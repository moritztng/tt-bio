import json, sys, statistics as st
b = json.load(open(sys.argv[1]))
print(b["stamp"])
rows = b["rows"]
for r in rows:
    s = r["block"]
    for k in ("dz", "dm"):
        if k in r:
            d, e = r[k], r[k + "_torch_bf16"]
            s += f"  {k} dev {d['rel_l2']:.4f}/{d['cos']:.5f} bf16 {e['rel_l2']:.4f} x{d['rel_l2']/e['rel_l2']:.2f}"
    if "permuted" in r:
        s += f"  PERM {[round(v['rel_l2'],3) for v in r['permuted'].values()]} ZERO {r['zero_seed_max_abs']}"
    print(s)
for k in ("dz", "dm"):
    rs = [r[k]["rel_l2"] / r[k + "_torch_bf16"]["rel_l2"] for r in rows if k in r]
    ab = [r[k]["rel_l2"] for r in rows if k in r]
    cs = [r[k]["cos"] for r in rows if k in r]
    print(k, "n", len(ab), "ratio med %.3f max %.3f" % (st.median(rs), max(rs)),
          "rel_l2 med %.4f max %.4f" % (st.median(ab), max(ab)), "cos min %.5f" % min(cs))
