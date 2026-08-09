#!/usr/bin/env python3
"""Are the fused operands wrong in situ, or is something downstream of them?

The kernel is bit-exact against the ttnn chain at both chunk widths when driven directly
(perf/megakernel/fused_gate_chanmajor.py), and the chunk width itself is bit-exact
(parity_isolate.py), yet the whole-trimul output inside the block is not. So the operands are
either wrong in situ -- different CB sizes, a different mask tensor, a reused descriptor --
or they are right and the divergence is downstream. This wraps fused_inputs and compares its
two operands against the ttnn chain computed from the very same gp_in_fused, on the first
chunk of the first trimul.

    TT_VISIBLE_DEVICES=3 python3 perf/megakernel/insitu_check.py --n 320
"""
import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--valid", type=int, default=298)
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
                           device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
    z0 = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.5, layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)

    orig = tt.trimul_fused.fused_inputs
    state = {"n": 0}

    def checked(gp, msk, ending, **kw):
        out_a, out_b = orig(gp, msk, ending, **kw)
        if state["n"] < 2:
            state["n"] += 1
            ca, cb, pa, pb = ttnn.chunk(gp, chunks=4, dim=-1)
            ra = ttnn.multiply(pa, ca, memory_config=DRAM,
                               input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            rb = ttnn.multiply(pb, cb, memory_config=DRAM,
                               input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            if msk is not None:
                ra = ttnn.multiply_(ra, ttnn.unsqueeze(msk, -1))
            pa_dims = (0, 3) + ((2, 1) if ending else (1, 2))
            pb_dims = (0, 3) + ((1, 2) if ending else (2, 1))
            ea = ttnn.to_torch(ttnn.permute(ra, pa_dims, memory_config=DRAM))
            eb = ttnn.to_torch(ttnn.permute(rb, pb_dims, memory_config=DRAM))
            ga, gb = ttnn.to_torch(out_a), ttnn.to_torch(out_b)
            print("  call %d ending=%s chunk_w=%d: a exact=%s (maxdiff %.3e) "
                  "b exact=%s (maxdiff %.3e)"
                  % (state["n"], ending, int(gp.shape[-1]) // 4,
                     torch.equal(ga, ea), (ga.float() - ea.float()).abs().max(),
                     torch.equal(gb, eb), (gb.float() - eb.float()).abs().max()), flush=True)
            for t in (ca, cb, pa, pb, ra, rb):
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        return out_a, out_b

    tt.trimul_fused.fused_inputs = checked
    tt._TRIMUL_FUSED = True
    print(f"\n=== in-situ operand check, N={N} ===", flush=True)
    ttnn.to_torch(layer.triangle_multiplication_start(z0, mask))
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
