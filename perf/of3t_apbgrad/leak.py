import json, torch
d = torch.load("/home/ttuser/of3t_apbgrad/tap_c64.pt", map_location="cpu", weights_only=False)
recs, meta = d["recs"], d["meta"]
out = {}
for blk in sorted(recs):
    P = [p for p in recs[blk] if any(o["g"] for r in p["ops"] for o in r["out"])][0]
    by = {r["idx"]: r for r in P["ops"]}
    G = lambda i, j=0: (sum(by[i]["out"][j]["g"][1:], by[i]["out"][j]["g"][0])
                        if by[i]["out"][j]["g"] else None)
    IN = lambda i: [x["val"] for x in by[i]["in"]]
    y, g = by[15]["out"][0]["val"], G(15)
    k = IN(10)[0]
    xin, xout = IN(14)[0], by[14]["out"][0]["val"]
    m = xin.abs() > 0
    f = float((xout[m] / xin[m]).median())
    rows = y.sum(-1, keepdim=True)
    inner = (g * y).sum(-1, keepdim=True)
    dl_f64 = y * (g - inner)
    dl_fix = y * (g - inner / rows)
    leak = dl_f64 - dl_fix                      # = y * inner * (1 - 1/sum y), the row-sum leak
    dq_f64, dq_fix, dq_leak = (dl_f64 * f) @ k, (dl_fix * f) @ k, (leak * f) @ k
    kbar = k.sum(-2)
    out[str(blk)] = {
        "leak_over_dlogits": float(leak.norm() / dl_fix.norm()),
        "dq_leak_over_dq_repaired": float(dq_leak.norm() / dq_fix.norm()),
        "dq_norm_unrepaired": float(dq_f64.norm()), "dq_norm_repaired": float(dq_fix.norm()),
        "amplification_dlogits_to_dq": float((dq_leak.norm() / dq_fix.norm())
                                             / (leak.norm() / dl_fix.norm())),
        "kbar_norm": float(kbar.norm()),
        "k_norm": float(k.norm())}
    o = out[str(blk)]
    print("blk %2d  leak/dlogits %.4f  dq_leak/dq_repaired %9.2f  amplification %8.1fx  "
          "||dq|| %.4e -> %.4e" % (blk, o["leak_over_dlogits"], o["dq_leak_over_dq_repaired"],
                                   o["amplification_dlogits_to_dq"], o["dq_norm_unrepaired"],
                                   o["dq_norm_repaired"]))
json.dump({"what": "the row-sum leak as a fraction of d_logits, and what q@k^T amplifies it to",
           "blocks": out},
          open("/home/ttuser/.coworker/wt/of3t-apbgrad/perf/of3t_apbgrad/LEAK_c64.json", "w"),
          indent=2)
