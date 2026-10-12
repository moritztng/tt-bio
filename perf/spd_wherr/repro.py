"""Reduce one wrong pixel of the Wormhole fp32-acc erratum to the fewest K terms that still produce it.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_wherr/repro.py --m 262144 --k 256 --n 768 --pixels 6 --out repro.json

Regenerates site_probe.py's draw 0 (seed 0: a N(0, 1), w N(0, 1/k), both bf16) on the host, finds the wrong pixels
of minimal_matmul's default config at HiFi4 by running it, and for each one:
1. isolates its dot product: a 32x32-tile problem holding only that row of a and that column of w, everything
   else zero. If that is still wrong, the defect needs nothing but these k products.
2. greedily zeroes one K term at a time, keeping each zeroing that leaves the pixel wrong, until no single term
   can go. What is left is a minimal trigger, printed with its products, the float64 sum and the device value.
Every case is also run at K block 1 (the clean config) as the control.
"""
import argparse, json
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=262144)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--pixels", type=int, default=6)
    ap.add_argument("--fid", default="HiFi4")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    A = torch.randn(a.m, a.k).bfloat16(); W = (torch.randn(a.k, a.n) / a.k ** 0.5).bfloat16()
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, a.fid),
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    k1 = ttnn.MinimalMatmulConfig(M_block_size=1, K_block_size=1, N_block_size=1, subblock_h=1, subblock_w=1,
                                  compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y))

    def mm(x, w, cfg=None):
        tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        tw = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        y = ttnn.experimental.minimal_matmul(input_tensor=tx, weight_tensor=tw, compute_kernel_config=ck,
                                             dtype=ttnn.bfloat16, config=cfg)
        out = ttnn.to_torch(y).double()
        for t in (tx, tw, y):
            ttnn.deallocate(t)
        return out

    def dot(av, wv, cfg=None):
        """av, wv: [k] bf16. The pixel at (0, 0) of a one-tile-by-one-tile problem."""
        x = torch.zeros(32, a.k, dtype=torch.bfloat16); x[0] = av
        w = torch.zeros(a.k, 32, dtype=torch.bfloat16); w[:, 0] = wv
        return float(mm(x, w, cfg)[0, 0])

    def wrong(ref, y):
        return abs(y - ref) > 0.25 and abs(y - ref) > 64 * 2.0 ** -8 * max(abs(ref), 2 ** -6)

    res = {"m": a.m, "k": a.k, "n": a.n, "fid": a.fid, "pixels": []}
    try:
        bad, step = [], 32768                      # find wrong pixels, a row block at a time
        for r0 in range(0, a.m, step):
            Y = mm(A[r0:r0 + step], W)
            R = A[r0:r0 + step].double() @ W.double()
            e = (Y - R).abs()
            for i in (e > 0.25).nonzero().tolist():
                bad.append((r0 + i[0], i[1]))
            if len(bad) >= a.pixels:
                break
        print(f"{len(bad)} wrong pixels found", flush=True)
        for r, c in bad[:a.pixels]:
            av, wv = A[r].clone(), W[:, c].clone()
            ref = float(av.double() @ wv.double())
            y_iso, y_k1 = dot(av, wv), dot(av, wv, k1)
            rec = dict(row=r, col=c, ref=ref, isolated=y_iso, k1=y_k1, reproduces=wrong(ref, y_iso))
            if rec["reproduces"]:
                keep = list(range(a.k))
                for i in sorted(range(a.k), key=lambda i: abs(float(av[i]) * float(wv[i]))):
                    trial = [j for j in keep if j != i]
                    m = torch.zeros(a.k, dtype=torch.bool); m[trial] = True
                    av2, wv2 = torch.where(m, av, 0), torch.where(m, wv, 0)
                    ref2 = float(av2.double() @ wv2.double())
                    if wrong(ref2, dot(av2, wv2)):
                        keep = trial
                m = torch.zeros(a.k, dtype=torch.bool); m[keep] = True
                av2, wv2 = torch.where(m, av, 0), torch.where(m, wv, 0)
                rec["minimal"] = dict(
                    k_index=keep, a=[float(av[j]) for j in keep], w=[float(wv[j]) for j in keep],
                    products=[float(av[j]) * float(wv[j]) for j in keep], k_tile=[j // 32 for j in keep],
                    ref=float(av2.double() @ wv2.double()), device=dot(av2, wv2), device_k1=dot(av2, wv2, k1))
            res["pixels"].append(rec); print(json.dumps(rec), flush=True)
    finally:
        ttnn.close_device(dev)
        a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
