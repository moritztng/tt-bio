import json, torch, sys
sys.path.insert(0, "/home/ttuser/.coworker/wt/of3t-apbgrad/perf/of3t_apbgrad")
d = torch.load("/home/ttuser/of3t_apbgrad/tap_c64.pt", map_location="cpu", weights_only=False)
refsite = torch.load("/home/ttuser/of3t_bwdaccum/refcot_f64_sites.pt",
                     map_location="cpu", weights_only=False)["site_cot"]
recs, meta = d["recs"], d["meta"]


def st(m, r):
    m, r = m.reshape(-1), r.reshape(-1)
    nr, nm = float(r.norm()), float(m.norm())
    return {"rel": float((m - r).norm() / (nr or 1.0)), "r": nm / nr if nr else float("inf"),
            "cos": float((m @ r) / (nm * nr)) if nm and nr else float("nan")}


out = {}
for blk in sorted(recs):
    M = meta[blk]; H, hd, D = M["n_heads"], M["head_dim"], M["padded_head_dim"]
    P = [p for p in recs[blk] if any(o["g"] for r in p["ops"] for o in r["out"])][0]
    by = {r["idx"]: r for r in P["ops"]}
    G = lambda i, j=0: (sum(by[i]["out"][j]["g"][1:], by[i]["out"][j]["g"][0])
                        if by[i]["out"][j]["g"] else None)
    def C(i, j=0):
        o = {}
        for oid, t in by[i]["out"][j]["contrib"]:
            o[oid] = t if oid not in o else o[oid] + t
        return o
    IN = lambda i: [x["val"] for x in by[i]["in"]]
    IID = lambda i: [x["id"] for x in by[i]["in"]]

    y = by[15]["out"][0]["val"]; g = G(15)
    dlog_dev = C(15)[IID(15)[0]]
    k, q = IN(10)[0], IN(11)[0]
    dq_dev = C(11)[IID(11)[0]]
    xin, xout = IN(14)[0], by[14]["out"][0]["val"]
    m = xin.abs() > 0
    f = float((xout[m] / xin[m]).median())
    rows = y.sum(-1, keepdim=True)
    inner = (g * y).sum(-1, keepdim=True)
    dlog_f64 = y * (g - inner)
    dlog_A = y * (g - inner / rows)                      # variant A: inner / sum(y)
    yhat = y / rows
    dlog_B = yhat * (g - (g * yhat).sum(-1, keepdim=True))   # variant B: renormalise y
    kbar = k.sum(-2)
    leak = torch.einsum("bhi,bhc->bhic", dlog_dev.sum(-1) * f, kbar)
    err = dq_dev - (dlog_f64 * f) @ k
    # proper regression of the error on the leak direction
    alpha = float((leak.reshape(-1) @ err.reshape(-1)) / (leak.reshape(-1).norm() ** 2))
    resid = err - alpha * leak
    B_, L, _ = P["s_in"].shape
    Wqkv = IN(0)[1]
    dv_dev = C(17)[IID(17)[1]]
    ds_g = C(25)[P["s_in_id"]]

    def ds_from(dl):
        dp = dl * f
        pack = torch.zeros(B_, L, 3, H, D, dtype=torch.float64)
        pack[:, :, 0] = (dp @ k).permute(0, 2, 1, 3)
        pack[:, :, 1] = (dp.transpose(-1, -2) @ q).permute(0, 2, 1, 3)
        pack[:, :, 2] = dv_dev.permute(0, 2, 1, 3)
        return pack.reshape(B_, L, 3 * H * D) @ Wqkv.transpose(-1, -2) + ds_g

    ref = refsite[f"blocks.{blk}.pre_norm_s"].reshape(P["s_in"].shape).to(torch.float64)
    out[str(blk)] = {
      "probs_row_sum_minus_one_rms": float(((rows - 1) ** 2).mean().sqrt()),
      "probs_row_sum_minus_one_max": float((rows - 1).abs().max()),
      "d_logits_row_sum_rms": {"card": float((dlog_dev.sum(-1) ** 2).mean().sqrt()),
                               "exact_bw_same_operands": float((dlog_f64.sum(-1) ** 2).mean().sqrt()),
                               "variant_A": float((dlog_A.sum(-1) ** 2).mean().sqrt()),
                               "variant_B": float((dlog_B.sum(-1) ** 2).mean().sqrt())},
      "dq_error_explained_by_the_leak": {
          "regression_coefficient": alpha,
          "leak_norm": float(leak.norm()), "error_norm": float(err.norm()),
          "residual_norm_after_removing_the_leak": float(resid.norm()),
          "fraction_of_the_error_variance_removed":
              1.0 - float(resid.norm() ** 2 / (err.norm() ** 2))},
      "module_ds_vs_reference": {
          "card": st(C(0)[P["s_in_id"]] + ds_g, ref),
          "exact_bw_same_operands": st(ds_from(dlog_f64), ref),
          "variant_A_inner_over_sum_y": st(ds_from(dlog_A), ref),
          "variant_B_renormalised_y": st(ds_from(dlog_B), ref)}}
    o = out[str(blk)]
    print("blk %2d rowsum(y)-1 rms %.3e | dlog rowsum card %.3e exact %.3e A %.2e B %.2e | "
          "leak explains %.4f of dq error | ds rel card %.4e exact %.4e A %.4e B %.4e"
          % (blk, o["probs_row_sum_minus_one_rms"], o["d_logits_row_sum_rms"]["card"],
             o["d_logits_row_sum_rms"]["exact_bw_same_operands"],
             o["d_logits_row_sum_rms"]["variant_A"], o["d_logits_row_sum_rms"]["variant_B"],
             o["dq_error_explained_by_the_leak"]["fraction_of_the_error_variance_removed"],
             o["module_ds_vs_reference"]["card"]["rel"],
             o["module_ds_vs_reference"]["exact_bw_same_operands"]["rel"],
             o["module_ds_vs_reference"]["variant_A_inner_over_sum_y"]["rel"],
             o["module_ds_vs_reference"]["variant_B_renormalised_y"]["rel"]))
json.dump({"what": "the softmax backward's zero-row-sum invariant, and what q@k^T does with "
                   "the leak", "blocks": out},
          open("/home/ttuser/.coworker/wt/of3t-apbgrad/perf/of3t_apbgrad/ROWSUM_c64.json", "w"),
          indent=2)
