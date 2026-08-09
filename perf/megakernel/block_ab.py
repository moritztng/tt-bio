#!/usr/bin/env python3
"""W4 milestone 5: the fused trimul input kernel, measured in the real Pairformer block.

Everything so far has been op-level. This runs the actual protenix-v2 layer-0 block at the
298 aa shape (N=320, c_z=256) with the kernel on and off in ONE process, so the two arms
share the device, the allocator and the timing discipline, and compares the block output
bit-exactly between arms. Both trimuls get the real padded pair mask (298 valid of 320),
which is what the trunk passes and what the op-level baseline had to include.

    TT_VISIBLE_DEVICES=3 python3 perf/megakernel/block_ab.py --n 320
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def pipe(dev, fn, warm=4, iters=7, k=6):
    """Median wall time of one call, k calls between one pair of syncs."""
    for _ in range(warm):
        r = fn()
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(k)]
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3 / k)
        for r in outs:
            if isinstance(r, ttnn.Tensor):
                ttnn.deallocate(r)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--valid", type=int, default=298)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    layer, c_z = build_layer(ckc)
    N = a.n
    torch.manual_seed(0)
    z_t = torch.randn(1, N, N, c_z) * 0.5
    tok = torch.zeros(1, N)
    tok[:, :a.valid] = 1
    pm = (tok[:, :, None] * tok[:, None, :])
    mask = ttnn.from_torch(pm, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    z0 = ttnn.from_torch(z_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tms = layer.triangle_multiplication_start
    tme = layer.triangle_multiplication_end
    print(f"\n=== Pairformer block, N={N} ({a.valid} aa valid), c_z={c_z}, card 3 ===",
          flush=True)

    rows, ref = [], {}
    for arm in ("baseline", "fused"):
        tt._TRIMUL_FUSED = (arm == "fused")
        # parity first, on a fresh call
        out_s = ttnn.to_torch(tms(z0, mask))
        out_e = ttnn.to_torch(tme(z0, mask))
        if arm == "baseline":
            ref["s"], ref["e"] = out_s, out_e
            note = "reference"
        else:
            note = ("trimul_start exact=%s trimul_end exact=%s"
                    % (torch.equal(out_s, ref["s"]), torch.equal(out_e, ref["e"])))
        ms_s = pipe(dev, lambda: tms(z0, mask))
        ms_e = pipe(dev, lambda: tme(z0, mask))
        used = tt._TRIMUL_FUSED and tt.trimul_fused.applicable(
            N, tt._trimul_chunk_size(N, tms._hidden),
            tt._triangle_mul_memory_config(N), tt._dtype(), tt._FAST_MODE)
        print("  [%-8s] trimul start %7.3f ms | end %7.3f ms | sum %7.3f ms | "
              "kernel active=%s  %s" % (arm, ms_s, ms_e, ms_s + ms_e, used, note), flush=True)
        rows.append(dict(arm=arm, trimul_start_ms=round(ms_s, 4), trimul_end_ms=round(ms_e, 4),
                         sum_ms=round(ms_s + ms_e, 4), kernel_active=bool(used), note=note))
        if arm == "fused":
            print("  program cache entries: %d, mask cache: %d"
                  % (len(tt.trimul_fused._PROGRAM_CACHE), len(tt.trimul_fused._MASK_CACHE)),
                  flush=True)
            rows.append(dict(arm="cache", programs=len(tt.trimul_fused._PROGRAM_CACHE),
                             masks=len(tt.trimul_fused._MASK_CACHE)))

    b, f = rows[0], [r for r in rows if r.get("arm") == "fused"][0]
    print("\n  two trimuls: %.3f -> %.3f ms = %.4fx" % (
        b["sum_ms"], f["sum_ms"], b["sum_ms"] / f["sum_ms"]), flush=True)
    # what that is worth on the whole block, using this run's own block time
    tt._TRIMUL_FUSED = False
    blk = pipe(dev, lambda: layer(None, ttnn.clone(z0))[1], warm=2, iters=5, k=3)
    print("  block (baseline, same run): %.3f ms -> projected with the kernel %.3f ms = "
          "%.4fx" % (blk, blk - (b["sum_ms"] - f["sum_ms"]),
                     blk / (blk - (b["sum_ms"] - f["sum_ms"]))), flush=True)
    rows.append(dict(arm="block_baseline_ms", ms=round(blk, 4),
                     projected_fused_ms=round(blk - (b["sum_ms"] - f["sum_ms"]), 4)))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(n=N, valid=a.valid, c_z=c_z, rows=rows),
                                         indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
