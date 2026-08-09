#!/usr/bin/env python3
"""Is the chunk width bit-exact, or is the kernel?

The in-model A/B is not bit-exact in either configuration tried: chunk width 64 with the
operands in DRAM, and chunk width 32 with them in L1. Those two differ in two ways at once
(operand memory and projection width), so neither run isolates the cause. This one does, by
running the UNCHANGED ttnn chain at chunk width 32 against itself at 64. If that alone is
inexact, the chunk width is the cause and the kernel is exonerated; the trimul docstring's
claim of bit-exactness at 32/64/128 then needs qualifying.

    TT_VISIBLE_DEVICES=3 python3 perf/megakernel/parity_isolate.py --n 320
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--valid", type=int, default=298)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer(ckc)
    N = a.n
    torch.manual_seed(0)
    tok = torch.zeros(1, N)
    tok[:, :a.valid] = 1
    mask = ttnn.from_torch(tok[:, :, None] * tok[:, None, :], layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    z0 = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.5, layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.bfloat16,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tms = layer.triangle_multiplication_start
    real_chunk = tt._trimul_chunk_size
    rows = []
    print(f"\n=== chunk-width parity, N={N}, {a.valid} aa valid ===", flush=True)
    ref = None
    for label, fused, width in (("ttnn chain, width 64 (production)", False, 64),
                                ("ttnn chain, width 32", False, 32),
                                ("fused kernel, width 32", True, 32)):
        tt._TRIMUL_FUSED = fused
        tt._trimul_chunk_size = lambda *_a, **_k: width
        out = ttnn.to_torch(tms(z0, mask))
        if ref is None:
            ref, note = out, "reference"
        else:
            d = (out.float() - ref.float()).abs()
            note = "exact=%s maxdiff=%.3e mean|d|=%.3e" % (
                torch.equal(out, ref), d.max(), d.mean())
        print("  %-34s %s" % (label, note), flush=True)
        rows.append(dict(arm=label, note=note))
    tt._trimul_chunk_size = real_chunk
    if a.out:
        Path(a.out).write_text(json.dumps(dict(n=N, rows=rows), indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
