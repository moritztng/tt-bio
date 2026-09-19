#!/usr/bin/env python3
"""D17: what a bijection REACHES, in the gradient's own currency.

A bijection covering 79 % of their tensors sounds like a claim about most of the gradient. It is
not. OF3's gradient mass is not spread evenly over 4,147 tensors -- one tensor carries a third of
it -- so "x of y tensors" and "p % of the squared norm" are different numbers and only the second
one says what a PROTOCOL 3d pass would be a pass over. K29 already insists on leaves against
total; this is the same rule denominated in gradient norm, and D17 is what it costs to skip it.

Everything below is computed on the published artifact, hash-verified first, in float64:

  * the squared L2 of every one of their 4,147 gradient tensors, and its share of the total;
  * the reach of `of3t-equivalence`'s tracer bijection (K22's in-scope set);
  * the reach of THIS row's device bijection, which built the shipped model on a card and placed
    by value, including the modules K22 could only list as out of scope;
  * the reach of the block-scope comparison instrument A actually ran;
  * and the unreached mass ranked by tensor, so the next lever is chosen by size rather than by
    how easy it looks.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

OUT = "perf/of3t_gradients"
BUNDLE = "/home/ttuser/of3t/bundle_min"
REF_BRANCH = "origin/wk/of3t-reference"
MANIFEST_GIT = "perf/of3t_reference/bundle_min/MANIFEST.json"
K22 = "perf/of3t_equivalence/bijection_manifest.json"
MAT64 = os.path.join(OUT, "full_model_of3_full_mat64_conf.json")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    import argparse
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default=MAT64,
                    help="the device-bijection artifact whose placed set defines our reach")
    a = ap.parse_args()
    mat_path = a.mat

    man = json.loads(subprocess.run(["git", "show", f"{REF_BRANCH}:{MANIFEST_GIT}"],
                                    capture_output=True, check=True).stdout)
    gfile = man["validated_gradient"]["file"]
    decl = {x["file"]: x for x in man["artifacts"] if "sha256" in x}
    got = sha256_file(os.path.join(BUNDLE, gfile))
    if got != decl[gfile]["sha256"]:
        raise SystemExit(f"{gfile}: sha256 {got} != manifest {decl[gfile]['sha256']}")

    g = torch.load(os.path.join(BUNDLE, gfile), map_location="cpu", weights_only=False)
    sq = {k: float(torch.linalg.vector_norm(v.to(torch.float64)) ** 2)
          for k, v in g.items() if v is not None}
    absent = [k for k, v in g.items() if v is None]
    del g
    total = sum(sq.values())
    rep = {"instrument": "D17: bijection reach in gradient norm, not tensor count",
           "reference": {"file": gfile, "sha256": got, "verified": True,
                         "manifest": f"{REF_BRANCH}:{MANIFEST_GIT}",
                         "num_recycles": man["validated_gradient"]["num_recycles"],
                         "published_global_norm":
                             man["validated_gradient"]["gradient_global_norm"]},
           "device_bijection_artifact": mat_path,
           "n_tensors": len(sq), "n_absent": len(absent),
           "total_squared_norm": total, "total_norm": total ** 0.5}
    print(f"{len(sq)} tensors, global norm {total**0.5:.12f} "
          f"(published {man['validated_gradient']['gradient_global_norm']:.12f})", flush=True)

    def top_level(k):
        return k.split(".", 1)[0]

    tl = {}
    for k, v in sq.items():
        tl.setdefault(top_level(k), [0, 0.0])
        tl[top_level(k)][0] += 1
        tl[top_level(k)][1] += v
    rep["by_top_level"] = {k: {"tensors": n, "squared_norm": s, "share": s / total,
                               "norm": s ** 0.5}
                           for k, (n, s) in sorted(tl.items(), key=lambda x: -x[1][1])}

    def reach(name, keys, note):
        keys = set(keys) & set(sq)
        s = sum(sq[k] for k in keys)
        d = {"tensors": len(keys), "of_tensors": len(sq), "tensor_share": len(keys) / len(sq),
             "squared_norm": s, "norm_share": s / total, "note": note}
        rep.setdefault("reach", {})[name] = d
        print(f"{name}: {len(keys)} of {len(sq)} tensors = {100*len(keys)/len(sq):.1f} % by "
              f"count, holding {100*s/total:.2f} % of the squared norm", flush=True)
        return d

    k22 = json.load(open(K22))["manifest"]
    reach("k22_tracer_bijection", k22.keys(),
          "of3t-equivalence's tracer through the CPU remap, K22's in-scope set")

    mat = json.load(open(mat_path))
    cannot = set(mat["presence"]["their_with_gradient_we_cannot_carry"])
    # The field used to be truncated to its first 60 entries while the count beside it said 610,
    # and reading the list instead of the count computed this reach over 60 missing tensors
    # rather than 610 -- D17's own defect arriving through the artifact instead of the metric.
    # The producer now emits the full set; this refuses anything else rather than trusting it.
    n_declared = mat["presence"]["n_their_with_gradient_we_cannot_carry"]
    if len(cannot) != n_declared:
        raise SystemExit(f"{mat_path}: unreached list has {len(cannot)} entries but declares "
                         f"{n_declared} -- a truncated list silently inflates every reach below")
    reach("device_bijection_mat64", set(sq) - cannot,
          "this row's bijection by value against the model BUILT ON A CARD, after a "
          "materialising forward at 64 tokens; the complement is the 610 their-tensors with a "
          "gradient that no device tensor carries")

    b0 = [k for k in sq if k.startswith("pairformer_stack.blocks.0.")]
    reach("instrument_a_block0_compared", b0,
          "all 57 of their tensors at pairformer block 0, the scope instrument A ran at")
    fused4 = {f"pairformer_stack.blocks.0.attn_pair_bias.mha.linear_{x}"
              for x in ("q.weight", "q.bias", "k.weight", "v.weight")}
    reach("instrument_a_block0_53", set(b0) - fused4,
          "the 53 actually compared; the other 4 live inside our fused qkv_weight and were "
          "reported absent rather than zero-filled")
    reach("pairformer_stack_all", [k for k in sq if k.startswith("pairformer_stack.")],
          "the whole 48-block trunk, the ceiling on any block-scope instrument")

    # Per pairformer block, because instrument A runs at block scope and A15 wants the share
    # of the set actually compared -- which is not the stack's 5.27 % divided by 48. The four
    # tensors inside our fused qkv are excluded from the compared column, as they are in the
    # instrument, so the two numbers describe the same set.
    fused4 = {"attn_pair_bias.mha.linear_q.weight", "attn_pair_bias.mha.linear_q.bias",
              "attn_pair_bias.mha.linear_k.weight", "attn_pair_bias.mha.linear_v.weight"}
    per_block = {}
    for k, v in sq.items():
        if not k.startswith("pairformer_stack.blocks."):
            continue
        i = k.split(".")[2]
        d = per_block.setdefault(i, {"tensors": 0, "squared_norm": 0.0,
                                     "compared_tensors": 0, "compared_squared_norm": 0.0})
        d["tensors"] += 1
        d["squared_norm"] += v
        if k.split(".", 3)[3] not in fused4:
            d["compared_tensors"] += 1
            d["compared_squared_norm"] += v
    for d in per_block.values():
        d["norm_share"] = d["squared_norm"] / total
        d["compared_norm_share"] = d["compared_squared_norm"] / total
    rep["per_pairformer_block"] = {k: per_block[k] for k in sorted(per_block, key=int)}

    unreached = sorted(((sq[k], k) for k in cannot & set(sq)), reverse=True)
    rep["unreached_by_device_bijection"] = {
        "tensors": len(unreached),
        "squared_norm": sum(v for v, _ in unreached),
        "norm_share": sum(v for v, _ in unreached) / total,
        "by_top_level": {},
        "top_20": [{"tensor": k, "squared_norm": v, "share": v / total}
                   for v, k in unreached[:20]]}
    for v, k in unreached:
        d = rep["unreached_by_device_bijection"]["by_top_level"].setdefault(
            top_level(k), {"tensors": 0, "squared_norm": 0.0})
        d["tensors"] += 1
        d["squared_norm"] += v
    for d in rep["unreached_by_device_bijection"]["by_top_level"].values():
        d["share"] = d["squared_norm"] / total

    rep["largest_single_tensors"] = [
        {"tensor": k, "squared_norm": v, "share": v / total,
         "reached_by_device_bijection": k not in cannot}
        for v, k in sorted(((v, k) for k, v in sq.items()), reverse=True)[:15]]

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "reach_by_norm.json")
    json.dump(rep, open(path, "w"), indent=1)
    print("\nby top level (share of squared norm):")
    for k, v in rep["by_top_level"].items():
        print(f"   {100*v['share']:6.2f} %  {v['tensors']:5d} tensors  L2 {v['norm']:.4f}  {k}")
    print("\nunreached by the device bijection, by top level:")
    for k, v in sorted(rep["unreached_by_device_bijection"]["by_top_level"].items(),
                       key=lambda x: -x[1]["squared_norm"]):
        print(f"   {100*v['share']:6.2f} %  {v['tensors']:5d} tensors  {k}")
    print("\nlargest single tensors:")
    for d in rep["largest_single_tensors"][:8]:
        print(f"   {100*d['share']:6.2f} %  reached={d['reached_by_device_bijection']}  "
              f"{d['tensor']}")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
