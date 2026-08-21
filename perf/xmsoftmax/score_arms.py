"""Score softmax arms on captured (logits, v) pairs, on the metric that survives the contraction.

Four arms, each run on device on the real logits at the site's own dtype, then contracted against
the real `v` on host in fp64:

  F        `ttnn.softmax` as shipped
  R        F renormalised by its own row sums -- 2 extra ops
  M        `_accurate_softmax`: max/subtract/exp/sum/divide -- 5 ops, the RF3 lever
  ceiling  a CORRECT softmax evaluated at the site's storage dtype, on host

`ceiling` is the number the port cannot beat by fixing the kernel, and it is not zero: at a bf16
site a correct bf16 softmax carries its own quantisation error, which Pass 1 measured can exceed
the buggy fp32 kernel's on peaked logits. An arm already at its ceiling has nothing left to win,
whatever its row deficit looks like.

The decision metric is rel_rms(o), not rel_rms(probs). Pass 1 measured that with a `v` independent
of `probs` the two are equal to three digits, so only real pairs can tell a model that will gain
from one that will not.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio.tenstorrent import _accurate_softmax  # noqa: E402

DT = {"DataType.FLOAT32": ttnn.float32, "DataType.BFLOAT16": ttnn.bfloat16}


def rel_rms(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="directory of pair_*.pt")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.pairs, "pair_*.pt")))
    if not files:
        raise SystemExit("no pairs in " + a.pairs)

    dev = ttnn.open_device(device_id=0)
    res = {"label": a.label, "pairs": []}
    try:
        for f in files:
            d = torch.load(f)
            lg, v = d["logits"], d["v"]
            dt = DT[d["in_dtype"]]
            lg64 = lg.to(torch.float64)
            v64 = v.to(torch.float64)
            p_ref = torch.softmax(lg64, dim=-1)
            o_ref = p_ref @ v64

            rec = {"file": os.path.basename(f), "site": d["site"], "shape": list(lg.shape),
                   "in_dtype": d["in_dtype"], "head_dim": v.shape[-1],
                   "logit_spread": float(lg.max() - lg.min()), "arms": {}}

            def score(name, p64):
                rs = p64.sum(-1)
                rec["arms"][name] = {
                    "rowsum_mean": float(rs.mean()), "rowsum_min": float(rs.min()),
                    "rel_rms_probs": rel_rms(p64, p_ref),
                    "rel_rms_o": rel_rms(p64 @ v64, o_ref),
                }

            # ceiling: a correct softmax, quantised to the dtype the site actually stores.
            store = torch.float32 if dt == ttnn.float32 else torch.bfloat16
            score("ceiling", torch.softmax(lg64, dim=-1).to(store).to(torch.float64))

            host = lg.to(torch.float32) if dt == ttnn.float32 else lg.to(torch.bfloat16)
            x = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
            fused = ttnn.softmax(x, dim=-1)
            score("F", ttnn.to_torch(fused).to(torch.float64))
            score("R", ttnn.to_torch(
                ttnn.divide(fused, ttnn.sum(fused, dim=-1, keepdim=True))).to(torch.float64))
            ttnn.deallocate(fused)
            score("M", ttnn.to_torch(_accurate_softmax(x)).to(torch.float64))
            ttnn.deallocate(x)

            fo = rec["arms"]["F"]["rel_rms_o"]
            for nm in ("R", "M", "ceiling"):
                r = rec["arms"][nm]["rel_rms_o"]
                rec["arms"][nm]["gain_vs_F"] = (fo / r) if r else None
            res["pairs"].append(rec)
            print("%-44s spread=%8.1f  " % (os.path.basename(f)[:44], rec["logit_spread"])
                  + "  ".join("%s=%.6f" % (n, rec["arms"][n]["rel_rms_o"])
                              for n in ("F", "R", "M", "ceiling")))
    finally:
        ttnn.close_device(dev)

    n = len(res["pairs"])
    res["summary"] = {
        nm: {
            "rel_rms_o": sum(p["arms"][nm]["rel_rms_o"] for p in res["pairs"]) / n,
            "rel_rms_probs": sum(p["arms"][nm]["rel_rms_probs"] for p in res["pairs"]) / n,
            "rowsum_mean": sum(p["arms"][nm]["rowsum_mean"] for p in res["pairs"]) / n,
        } for nm in ("F", "R", "M", "ceiling")
    }
    s = res["summary"]
    for nm in ("R", "M", "ceiling"):
        s[nm]["gain_vs_F"] = (s["F"]["rel_rms_o"] / s[nm]["rel_rms_o"]
                              if s[nm]["rel_rms_o"] else None)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("\n== %s (mean over %d pairs) ==" % (a.label, n))
    for nm in ("F", "R", "M", "ceiling"):
        g = s[nm].get("gain_vs_F")
        print("  %-8s rel_rms_o=%.6f  rowsum=%.6f%s"
              % (nm, s[nm]["rel_rms_o"], s[nm]["rowsum_mean"],
                 ("  gain vs F=%.2fx" % g) if g else ""))


if __name__ == "__main__":
    main()
