#!/usr/bin/env python3
"""Price every ladder arm's accuracy against a float64 host reference, never against ttnn.

The comparison that matters is arm-vs-truth, not arm-vs-other-arm: two device arms can agree to
the digit and both be wrong, and the shipped arm is an approximation too
(`bit-exactness-not-required-accuracy-bar-is`, STANDING). So the reference is the same operands
contracted in float64 on the host, and both the derived config and every candidate are scored
against it with the same metrics. `torch.equal` between the two device arms is recorded beside
that, because bit-exactness is the cheapest regression signal where it happens to be free -- but
it is not the bar.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock"))

from derive import build, derive                                              # noqa: E402
from ladder import SHAPES                                                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="SW1,SW3,DIT,KA,KB")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    want = set(a.shapes.split(","))

    import torch
    import ttnn
    from tt_bio import tenstorrent as T

    device = T.get_device()
    GX, GY = T.COMPUTE_GRID_MAIN
    kcls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    KC = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    MC = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}
    torch.manual_seed(0)
    out = []

    for nm, b, M, K, N, dt, r0, ro, share in SHAPES:
        if nm not in want:
            continue
        tdt = ttnn.bfloat16 if dt == "bf16" else ttnn.float32
        mt, kt, nt = b * (M // 32), K // 32, N // 32
        c = derive(mt, kt, nt, GX, GY, True, M, K, N)
        c["fp32_dest_acc"] = True
        xt = torch.randn(1, b, M, K) * 0.05
        wt = torch.randn(K, N) * 0.05
        # the host reference is float64 over the operands AS THE DEVICE HOLDS THEM, so the
        # round-trip to bf16 is not scored as a device error
        x = ttnn.from_torch(xt, dtype=tdt, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=MC[r0])
        w = ttnn.from_torch(wt, dtype=tdt, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ref = (ttnn.to_torch(x).double() @ ttnn.to_torch(w).double()).squeeze(0)

        def score(pc):
            kw = dict(compute_kernel_config=KC, memory_config=MC[ro], dtype=tdt)
            if pc is None:
                kw["core_grid"] = ttnn.CoreGrid(x=GX, y=GY)
            else:
                kw["program_config"] = pc
            r = ttnn.linear(x, w, **kw)
            h = ttnn.to_torch(r).squeeze(0).double()
            ttnn.deallocate(r)
            d = (h - ref).abs()
            pcc = torch.corrcoef(torch.stack([h.flatten(), ref.flatten()]))[0, 1].item()
            return h, {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                       "pcc": round(pcc, 10)}

        base_h, base_m = score(None)
        cands = {"mirror": build(ttnn, c)}
        for h in (1, 2, c["per_core_M"] // 2):
            if h and h > 0 and c["per_core_M"] % h == 0 and h != c["out_block_h"]:
                cands[f"obh{h:02d}"] = build(ttnn, c, out_block_h=h)
        for iw in (1, 2, 4, 8, 16, 32):
            if iw != c["in0_block_w"] and kt % iw == 0:
                cands[f"ibw{iw:02d}"] = build(ttnn, c, in0_block_w=iw)
        rows = {"derived": {**base_m, "equal_to_derived": True}}
        for tag, pc in cands.items():
            try:
                h, m = score(pc)
            except Exception as e:                                             # noqa: BLE001
                rows[tag] = {"refused": repr(e)[:90]}
                continue
            rows[tag] = {**m, "equal_to_derived": bool(torch.equal(h, base_h))}
        print(f"\n== {nm} b={b} M={M} K={K} N={N} {dt}  {c['cls']} pcm={c['per_core_M']} "
              f"ibw={c['in0_block_w']} obh={c['out_block_h']}", flush=True)
        for tag, m in rows.items():
            if "refused" in m:
                print(f"   {tag:10s} refused", flush=True)
                continue
            print(f"   {tag:10s} max {m['max_abs']:.10e}  mean {m['mean_abs']:.6e}  "
                  f"pcc {m['pcc']:.10f}  bitexact={m['equal_to_derived']}", flush=True)
        out.append({"shape": nm, "derived": c, "rows": rows})
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
