#!/usr/bin/env python3
"""of3t-inproj: what fullstep64's score.py does not report, on the same bijection and arithmetic.

    score_inproj.py --f64 grads_f64.pt --bf16 grads_bf16.pt --bijection BIJECTION.json \
        --shapes DEVICE_SHAPES.json --on INON=g1.pt --aa INON2=g2.pt --pad PADON=g3.pt --out F.json

  in-projection  per upstream weight kind (linear_a_p, linear_b_p, linear_a_g, linear_b_g), pooled
                 per stack and over all stacks, rel vs float64 with upstream bf16 beside it, plus
                 the worst single tensor of each kind;
  layer_norm_in  the trimul input LayerNorm, whose gradient reaches it only through the
                 in-projection's input cotangent: the dx question;
  A/A            ON against its repeat, tensor by tensor, bit-identical count;
  pad control    PADON against ON: bit-identical count, and the confidence and global rel of both.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_fullstep64"))
from score import head_of, load_device, to_upstream, triple  # noqa: E402

KINDS = ("linear_a_p", "linear_b_p", "linear_a_g", "linear_b_g")
STACK = (("aux_heads.pairformer_embedding.", "confidence_pairformer"),
         ("pairformer_stack.", "pairformer_stack"), ("msa_module.", "msa_module"),
         ("template_embedder.", "template_embedder"))


def stack_of(k):
    return next((s for p, s in STACK if k.startswith(p)), "other")


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--f64", "--bf16", "--bijection", "--shapes", "--on", "--aa", "--pad", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    ref = {k: v.to(torch.float64) for k, v in torch.load(a.f64, weights_only=False).items()
           if v is not None and bool(v.any())}
    bf = {k: v.to(torch.float64) for k, v in torch.load(a.bf16, weights_only=False).items()
          if v is not None}
    bij = json.loads(Path(a.bijection).read_text())
    shapes = json.loads(Path(a.shapes).read_text())["shapes"]
    placed = set(bij["placements"]) | set(bij["derived"])
    total = sum(float((v * v).sum()) for v in ref.values())

    raw, up = {}, {}
    for spec in (a.on, a.aa, a.pad):
        name, path = spec.split("=", 1)
        raw[name] = load_device(path, shapes)
        up[name] = to_upstream(raw[name], bij)[0]
    on, aa, pad = (s.split("=", 1)[0] for s in (a.on, a.aa, a.pad))

    def rel(arm, keys):
        keys = [k for k in keys if k in ref]
        t = triple([(ref[k], arm.get(k, torch.zeros_like(ref[k]))) for k in keys])
        if t is None:
            return None
        return {"n": len(keys), "rel": t["rel"], "r": t["r"], "cos": t["cos"],
                "mass_fraction": t["ref_sq"] / total,
                "empty": sum(1 for k in keys if k not in arm),
                "unplaced": sum(1 for k in keys if k not in placed)}

    rep = {"reference": a.f64, "in_projection": {}, "layer_norm_in": {}}
    inproj = [k for k in ref if re.search(r"tri_mul_(out|in)\.linear_[ab]_[pg]\.weight$", k)]
    rep["in_projection"]["all"] = {"ON": rel(up[on], inproj), "BF16": rel(bf, inproj)}
    for kind in KINDS:
        ks = [k for k in inproj if f".{kind}." in k]
        row = {"all": {"ON": rel(up[on], ks), "BF16": rel(bf, ks)}}
        for st in sorted({stack_of(k) for k in ks}):
            sk = [k for k in ks if stack_of(k) == st]
            row[st] = {"ON": rel(up[on], sk), "BF16": rel(bf, sk)}
        per = {k: rel(up[on], [k])["rel"] for k in ks}
        w = max(per, key=per.get)
        row["worst"] = {"tensor": w, "ON": per[w], "BF16": rel(bf, [w])["rel"]}
        rep["in_projection"][kind] = row
    lnin = [k for k in ref if re.search(r"tri_mul_(out|in)\.layer_norm_in\.(weight|bias)$", k)]
    rep["layer_norm_in"] = {"ON": rel(up[on], lnin), "BF16": rel(bf, lnin),
                            "trunk_ON": rel(up[on], [k for k in ref if head_of(k) == "trunk"
                                                     and k in placed])}
    rep["placed_but_empty_ON"] = sorted(k for k in ref if k in placed and k not in up[on])
    same = lambda x, y: sum(1 for k in raw[x] if k in raw[y] and torch.equal(raw[x][k], raw[y][k]))  # noqa: E731
    rep["aa"] = {"pair": [on, aa], "n": len(raw[on]), "bit_identical": same(on, aa),
                 "keysets_equal": set(raw[on]) == set(raw[aa])}
    conf = [k for k in ref if head_of(k) == "confidence" and k in placed]
    glob = [k for k in ref if k in placed]
    rep["pad_control"] = {
        "pair": [on, pad], "n": len(raw[on]), "bit_identical": same(on, pad),
        "moved": sorted(k for k in raw[on] if k in raw[pad] and not torch.equal(raw[on][k], raw[pad][k]))[:20],
        "confidence_rel": {on: rel(up[on], conf)["rel"], pad: rel(up[pad], conf)["rel"]},
        "global_rel": {on: rel(up[on], glob)["rel"], pad: rel(up[pad], glob)["rel"]}}
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"in_projection": rep["in_projection"]["all"],
                      "layer_norm_in": rep["layer_norm_in"], "empty": len(rep["placed_but_empty_ON"]),
                      "aa": rep["aa"], "pad": {k: v for k, v in rep["pad_control"].items() if k != "moved"}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
