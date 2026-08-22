#!/usr/bin/env python3
"""Is ttnn.softmax's row deficit ONE scalar per row, and if so, is that scalar a reciprocal?

The row-sum probe says every row of `ttnn.softmax` comes out ~0.46 % short, at every S in
320..1024 and at every row peakedness (temp 0.25/1/4). Flat in peakedness means it is not an
accumulation error. This checks the remaining claim directly: that the kernel returns
`exps / Z'` for a single per-row Z', i.e. one scalar, and then reports Z'/Z.

If ratio_spread_within_row is at the fp32 noise floor, the error is EXACTLY one scalar per row and
the only candidate left is the normalising divide. Z'/Z > 1 means the kernel's denominator is too
large, which for a recip-and-multiply kernel means the reciprocal it applied was too small.
"""
import sys, json
from pathlib import Path
ROOT = Path("/home/ttuser/.coworker/wt/sdpa-rowsum-normalisation-kernel-fix")
sys.path.insert(0, str(ROOT))
import torch, ttnn
import tt_bio.tenstorrent as T
assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__

dev = T.get_device()
out = {}
for S in (320, 512):
    for temp in (1.0, 4.0):
        torch.manual_seed(0)
        B, H = 4, 4
        # scores straight in fp32: no matmul, no bias, nothing between the input and the softmax
        sc_h = (torch.randn(B, H, S, S) * temp)
        sc = ttnn.from_torch(sc_h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        p = ttnn.to_torch(ttnn.softmax(sc, dim=-1)).double()
        pa = ttnn.to_torch(T._accurate_softmax(sc)).double()
        ttnn.deallocate(sc)

        # the exps the kernel must have formed, in fp64, from the same scores
        d = sc_h.double() - sc_h.double().max(dim=-1, keepdim=True).values
        e = d.exp()
        Z = e.sum(-1, keepdim=True)

        for nm, pp in (("softmax", p), ("accurate", pa)):
            # p_i = e_i / Z'  =>  e_i / p_i = Z' for every i in the row
            r = e / pp.clamp_min(1e-300)
            # only columns that actually carry weight can measure the scalar
            live = (pp > 1e-6)
            rl = torch.where(live, r, torch.nan)
            zp = torch.nanmedian(rl, dim=-1).values
            rel = (rl / zp.unsqueeze(-1))
            spread = torch.nanquantile(rel.flatten().float()[~rel.flatten().isnan()], 
                                       torch.tensor([0.5, 0.999])).tolist()
            zratio = (zp / Z.squeeze(-1))
            out[f"S{S}_t{temp:g}_{nm}"] = {
                "Zprime_over_Z_mean": zratio.mean().item(),
                "Zprime_over_Z_min": zratio.min().item(),
                "Zprime_over_Z_max": zratio.max().item(),
                "frac_rows_Zprime_gt_Z": (zratio > 1).double().mean().item(),
                # how constant the scalar is WITHIN a row: 1.0 means exactly one scalar
                "within_row_ratio_median": spread[0],
                "within_row_ratio_p999": spread[1],
                "rowsum": pp.sum(-1).mean().item() - 1.0,
                "live_cols_per_row": live.double().sum(-1).mean().item(),
            }
            s = out[f"S{S}_t{temp:g}_{nm}"]
            print(f"S{S} t{temp:g} {nm:9s} Z'/Z={s['Zprime_over_Z_mean']:.8f} "
                  f"[{s['Zprime_over_Z_min']:.6f},{s['Zprime_over_Z_max']:.6f}] "
                  f"frac>1={s['frac_rows_Zprime_gt_Z']:.3f} "
                  f"within-row p999={s['within_row_ratio_p999']:.8f} "
                  f"rowsum dev={s['rowsum']:+.6f} live={s['live_cols_per_row']:.0f}", flush=True)
Path("/tmp/scalar_probe.json").write_text(json.dumps(out, indent=1))
T.cleanup()
