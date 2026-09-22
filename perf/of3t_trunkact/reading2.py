#!/usr/bin/env python3
"""of3t-trunkact: our device trunk against the TWO ends of upstream's own bf16 policy axis.

`bf16auto` is upstream's training recipe and the reference the GRADIENTS clause compares to; it
keeps layer norms, softmax and reductions in float32 because that is what `torch.autocast` does.
`bf16full` runs every one of those in bf16, which is what a bf16 device kernel implements. The
question this answers is whether the trunk's residual is an OP or the POLICY."""
from __future__ import annotations
import argparse, json, statistics as st

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.walk))
    rows = d["per_block"]
    n = len(rows)
    arms = list(d["arms"])

    def s(arm, leaf, scope="real"):
        return [r[scope][f"{arm}_{leaf}"]["abs_err"] for r in rows]

    R = {"what": __doc__.strip().splitlines()[0], "walk": a.walk, "host": d["host"],
         "tree": d["tree"], "tokens": d["tokens"], "real_tokens": d["real_tokens"],
         "cotangent_pad_control": d["cotangent_pad_control"],
         "controls": {
             "ref64_final": d["final_ref64"], "bf16auto_final": d["final_bf16"],
             "full16_final": d["final_full16"],
             "frame_REF_F64_N384": {"s_norm": 2547074.6682744888,
                                    "z_norm": 15590710.480237307},
             "frame_REF_BF16AUTO_N384": {"s_norm": 2545792.434019603,
                                         "z_norm": 15590158.54011093}},
         "policy_axis": {}, "per_block_ratio": {}, "block47": {}}
    R["controls"]["ref64_reproduces_frame"] = (
        d["final_ref64"]["s_norm"] == 2547074.6682744888
        and d["final_ref64"]["z_norm"] == 15590710.480237307)
    R["controls"]["bf16auto_reproduces_frame"] = (
        d["final_bf16"]["s_norm"] == 2545792.434019603
        and d["final_bf16"]["z_norm"] == 15590158.54011093)

    for leaf in ("s", "z"):
        for scope in ("real", "padded"):
            auto = s("bf16", leaf, scope)
            full = s("full16", leaf, scope)
            for arm in arms:
                o = s(arm, leaf, scope)
                R["per_block_ratio"][f"{arm}_{leaf}_{scope}_over_bf16auto"] = [
                    (o[i] / auto[i]) if auto[i] else None for i in range(n)]
                R["per_block_ratio"][f"{arm}_{leaf}_{scope}_over_full16"] = [
                    (o[i] / full[i]) if full[i] else None for i in range(n)]
            R["per_block_ratio"][f"full16_{leaf}_{scope}_over_bf16auto"] = [
                (full[i] / auto[i]) if auto[i] else None for i in range(n)]

    for leaf in ("s", "z"):
        for scope in ("real", "padded"):
            auto, full = s("bf16", leaf, scope)[-1], s("full16", leaf, scope)[-1]
            ent = {"bf16auto_abs": auto, "full16_abs": full,
                   "policy_factor_full_over_auto": full / auto if auto else None}
            for arm in arms:
                o = s(arm, leaf, scope)[-1]
                ent[f"{arm}_abs"] = o
                ent[f"{arm}_over_bf16auto"] = o / auto if auto else None
                ent[f"{arm}_over_full16"] = o / full if full else None
            R["block47"][f"{leaf}_{scope}"] = ent

    for k, v in list(R["per_block_ratio"].items()):
        vv = [x for x in v if x is not None]
        R["policy_axis"][k] = {"min": min(vv), "median": st.median(vv), "max": max(vv),
                               "at_block47": v[-1]}

    json.dump(R, open(a.out, "w"), indent=2)
    print("block 47, REAL tokens only, absolute error against the same float64:")
    for k in ("s_real", "z_real", "s_padded", "z_padded"):
        e = R["block47"][k]
        print(f"  {k:9s} bf16auto={e['bf16auto_abs']:14.2f} full16={e['full16_abs']:14.2f} "
              f"(policy {e['policy_factor_full_over_auto']:6.3f}x)  "
              f"ship={e['ship_abs']:14.2f} ship/auto={e['ship_over_bf16auto']:7.3f} "
              f"ship/full={e['ship_over_full16']:7.3f}")
    print("\nper-block ratio, min / median / max / at 47:")
    for k in sorted(R["policy_axis"]):
        if "_real_" not in k:
            continue
        v = R["policy_axis"][k]
        print(f"  {k:44s} {v['min']:8.3f} {v['median']:8.3f} {v['max']:8.3f} {v['at_block47']:8.3f}")
    print("\ncotangent pad control:", json.dumps(R["cotangent_pad_control"]))
    print("controls reproduce frame:", R["controls"]["ref64_reproduces_frame"],
          R["controls"]["bf16auto_reproduces_frame"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
