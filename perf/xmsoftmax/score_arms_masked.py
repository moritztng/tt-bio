"""Score softmax arms at a masked site, separating degenerate (fully-masked) rows.

Fp32TriangleAttention's [I,1,1,J] pair mask zeroes whole rows: 27% of rows at 4644 have every
entry at -1e9. A correct softmax makes those rows uniform. _accurate_softmax cannot, because
ttnn.max rounds its fp32 input to bf16 (-1e9 -> -9.98244e8, exact bf16 truncation), so the
subtraction leaves -1.76e6 instead of 0, exp underflows, the row sums to 0 and divide gives nan.

Two guards, both measured rather than assumed:
  Md  clamp d from below at -80 before exp: one extra op on the FULL score tensor. A fully
      masked row becomes uniform, matching the reference exactly; a normal row is perturbed by
      exp(-80)=1.8e-35, which is 1e-35 relative on the row sum.
  Ms  clamp the row sum from below: one extra op on the [.., 1] reduction, near-free. A fully
      masked row comes out all-zero instead of uniform.
"""
import argparse, glob, json, os, sys
sys.path.insert(0, "/home/ttuser/.coworker/wt/accurate-softmax-crossmodel")
import torch, ttnn
from tt_bio.tenstorrent import _accurate_softmax

DT = {"DataType.FLOAT32": ttnn.float32, "DataType.BFLOAT16": ttnn.bfloat16}


def rel_rms(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def chain(x, ckc=None, floor_d=None, floor_s=None):
    m = ttnn.max(x, dim=-1, keepdim=True)
    d = ttnn.subtract(x, m)
    ttnn.deallocate(m)
    if floor_d is not None:
        d = ttnn.maximum(d, floor_d)
    ttnn.exp(d, output_tensor=d)
    s = ttnn.sum(d, dim=-1, keepdim=True, compute_kernel_config=ckc)
    if floor_s is not None:
        s = ttnn.maximum(s, floor_s)
    p = ttnn.divide(d, s)
    ttnn.deallocate(d); ttnn.deallocate(s)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.pairs, "pair_*.pt")))
    dev = ttnn.open_device(device_id=0)
    res = {"label": a.label, "pairs": []}
    try:
        for f in files:
            d = torch.load(f)
            lg, v = d["logits"], d["v"]
            dt = DT[d["in_dtype"]]
            lg64, v64 = lg.to(torch.float64), v.to(torch.float64)
            p_ref = torch.softmax(lg64, dim=-1)
            o_ref = p_ref @ v64
            # A row is degenerate when every entry sits at the mask value.
            deg = (lg.max(-1).values <= -1e8)
            keep = ~deg
            rec = {"file": os.path.basename(f), "site": d["site"], "shape": list(lg.shape),
                   "in_dtype": d["in_dtype"],
                   "n_rows": int(deg.numel()), "n_degenerate": int(deg.sum()),
                   "logit_spread": float(lg.max() - lg.min()), "arms": {}}

            def score(name, p64):
                o = p64 @ v64
                rec["arms"][name] = {
                    "rowsum_mean": float(p64.sum(-1).mean()),
                    "rel_rms_o_all": rel_rms(o, o_ref),
                    "rel_rms_o_real": rel_rms(o[keep], o_ref[keep]),
                    "n_nan": int(torch.isnan(p64).sum()),
                }

            store = torch.float32 if dt == ttnn.float32 else torch.bfloat16
            score("ceiling", torch.softmax(lg64, -1).to(store).to(torch.float64))
            host = lg.to(torch.float32) if dt == ttnn.float32 else lg.to(torch.bfloat16)
            x = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
            fused = ttnn.softmax(x, dim=-1)
            score("F", ttnn.to_torch(fused).to(torch.float64))
            ttnn.deallocate(fused)
            score("M", ttnn.to_torch(_accurate_softmax(x)).to(torch.float64))
            score("Md", ttnn.to_torch(chain(x, floor_d=-80.0)).to(torch.float64))
            score("Ms", ttnn.to_torch(chain(x, floor_s=1e-30)).to(torch.float64))
            ttnn.deallocate(x)
            fa = rec["arms"]["F"]
            for nm in ("M", "Md", "Ms", "ceiling"):
                for k in ("all", "real"):
                    r = rec["arms"][nm]["rel_rms_o_" + k]
                    rec["arms"][nm]["gain_" + k] = (fa["rel_rms_o_" + k] / r) if r else None
            res["pairs"].append(rec)
            print("%s  rows=%d degenerate=%d" % (rec["file"][:46], rec["n_rows"], rec["n_degenerate"]))
            for nm in ("F", "M", "Md", "Ms", "ceiling"):
                A = rec["arms"][nm]
                print("   %-8s all=%.6f real=%.6f rowsum=%.6f nan=%d" %
                      (nm, A["rel_rms_o_all"], A["rel_rms_o_real"], A["rowsum_mean"], A["n_nan"]))
    finally:
        ttnn.close_device(dev)
    n = len(res["pairs"])
    res["summary"] = {nm: {k: sum(p["arms"][nm][k] for p in res["pairs"]) / n
                           for k in ("rel_rms_o_all", "rel_rms_o_real")}
                      for nm in ("F", "M", "Md", "Ms", "ceiling")}
    for nm in ("M", "Md", "Ms", "ceiling"):
        for k in ("all", "real"):
            r = res["summary"][nm]["rel_rms_o_" + k]
            res["summary"][nm]["gain_" + k] = (
                res["summary"]["F"]["rel_rms_o_" + k] / r) if r else None
    json.dump(res, open(a.out, "w"), indent=2)
    print("\n== %s (mean over %d pairs) ==" % (a.label, n))
    for nm in ("F", "M", "Md", "Ms", "ceiling"):
        s = res["summary"][nm]
        print("  %-8s all=%.6f (%sx)  real=%.6f (%sx)" % (
            nm, s["rel_rms_o_all"],
            ("%.2f" % s["gain_all"]) if s.get("gain_all") else "-",
            s["rel_rms_o_real"],
            ("%.2f" % s["gain_real"]) if s.get("gain_real") else "-"))


if __name__ == "__main__":
    main()
