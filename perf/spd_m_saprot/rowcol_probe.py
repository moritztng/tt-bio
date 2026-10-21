"""Where normal mode's isolated wrong fc1 elements come from (fc1_check, seed 0: [2,447,2411] device -0.0601 against
2.121, the same value from the 2D program and minimal_matmul). Rebuilds fc1_check's x and w, takes the 32-row x tile and
the 32-column w tile around each element, and runs x_tile @ w_tile on ttnn.matmul at every fidelity, fp32 dest on/off,
packer L1 acc on/off; then per 32-wide K chunk (one tile product each) against float64, to name the chunk that is off.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/rowcol_probe.py --out probe.jsonl
"""
import argparse
import json
import math

import torch
import ttnn

ELEMS = ((2, 447, 2411), (1, 46, 829), (3, 228, 1916))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tt_bio import tenstorrent as T
    T.set_fast_mode(False)
    dev = T.get_device()
    d, ff, B, L = 1280, 5120, 7, 544
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(B, L, d, generator=gen).to(torch.bfloat16)
    w = (torch.randn(d, ff, generator=gen) / math.sqrt(d) * 2).to(torch.bfloat16)
    out = open(a.out, "a")

    def ck(fid, f32, pl1):
        return ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                                fp32_dest_acc_en=f32, packer_l1_acc=pl1)

    def mm(xt, wt, c):
        xd = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wd = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        o = ttnn.to_torch(ttnn.matmul(xd, wd, compute_kernel_config=c, dtype=ttnn.float32)).double()
        ttnn.deallocate(xd), ttnn.deallocate(wd)
        return o

    for bi, r, col in ELEMS:
        r0, c0 = r - r % 32, col - col % 32
        xt, wt = x[bi, r0:r0 + 32, :].contiguous(), w[:, c0:c0 + 32].contiguous()
        ex = xt.double() @ wt.double()
        i, j = r - r0, col - c0
        rec = dict(elem=[bi, r, col], exact=float(ex[i, j]))
        # the full page row block as fc1 sees it, plain ttnn.matmul (no bias, no GELU)
        for fid in ("HiFi4", "HiFi3", "HiFi2", "LoFi"):
            for f32 in (True, False):
                for pl1 in (True, False):
                    o = mm(xt, wt, ck(fid, f32, pl1))
                    e = (o - ex).abs()
                    rec[f"{fid}_{int(f32)}{int(pl1)}"] = [float(o[i, j]), float(e.max()), int((e > 0.25).sum())]
        # per K chunk, HiFi4 fp32 dest: one tile product each
        bad = []
        for k in range(0, d, 32):
            o = mm(xt[:, k:k + 32].contiguous(), wt[k:k + 32].contiguous(), ck("HiFi4", True, False))
            exk = xt[:, k:k + 32].double() @ wt[k:k + 32].double()
            e = (o - exk).abs()
            if float(e.max()) > 0.05:
                ii, jj = divmod(int(e.argmax()), 32)
                bad.append(dict(k=k, max=float(e.max()), at=[ii, jj], dev=float(o[ii, jj]), exact=float(exk[ii, jj]),
                                n=int((e > 0.05).sum()),
                                x=[float(v) for v in xt[ii, k:k + 32]], w=[float(v) for v in wt[k:k + 32, jj]]))
        rec["chunks_bad"] = bad
        print(json.dumps({k: v for k, v in rec.items() if k != "chunks_bad"}), len(bad), flush=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
