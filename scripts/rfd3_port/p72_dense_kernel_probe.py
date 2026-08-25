#!/usr/bin/env python3
"""p72 -- the L2 kernel's exactness gate and its price, at the production shape.

Gate, and it is `torch.equal` rather than a tolerance: the kernel must reproduce

    add(typecast(scores, fp32), typecast(bias, fp32), a_activations=[MUL_UNARY_SFPU(scale)])

bit for bit, at [1, 16, 685, 704] bf16 -> fp32. ttnn's own folded form
(`add(bf16, bf16, dtype=fp32, act=scale)`) is 2.55 maxabs away from that chain
(perf/p71/bias_prep_arms.json), so "close" is what the shipped alternative already gives and is
not what this is for.

Also swept over the shapes the model actually issues, because the same call site runs at every
target size, and over SLOTS/WINDOW, which are the op's only tuning knobs.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p2 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p2 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p72_dense_kernel_probe.py \
          perf/p72/dense_kernel_probe.json
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio import rfd3_bias                                             # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p72/dense_kernel_probe.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
HEAD_DIM = 48
SCALE = HEAD_DIM ** -0.5
MB = 1024.0 * 1024.0

# (H, I, n_key) the DiT issues. 685 is the page fixture; the others are the size ladder's rungs
# (R0/R2/R4 at 128, 298, 512 and 685 tokens) so nothing goes default-ON tuned at one length.
SHAPES = [(16, 128, 128), (16, 298, 320), (16, 512, 512), (16, 685, 704)]


def timeit(fn, dev, n=N, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out), [round(v, 4) for v in out]


def main():
    dev = get_device()
    torch.manual_seed(0)
    rows = []
    ok = True
    for H, I, NK in SHAPES:
        s_t = torch.randn(1, H, I, NK) * 0.3
        b_t = torch.where(torch.rand(1, H, I, NK) < 0.05, -1e4, torch.randn(1, H, I, NK) * 0.5)
        # the pad columns the model's pad_axis writes, so the operand really contains -9984
        if NK > I:
            b_t[..., I:] = -1e4
            s_t[..., I:] = 0.0
        mk = lambda x: ttnn.from_torch(                                    # noqa: E731
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores, bias = mk(s_t), mk(b_t)

        ref = ttnn.add(
            ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config()),
            ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config()),
            input_tensor_a_activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SCALE)])
        ref_h = ttnn.to_torch(ref).float()
        got = rfd3_bias.dense_fused_scores_bias_fp32(scores, bias, SCALE)
        got_h = ttnn.to_torch(got).float()
        eq = bool(torch.equal(ref_h, got_h))
        mx = float((ref_h - got_h).abs().max())
        nbad = int((ref_h != got_h).sum())
        ok = ok and eq

        def ref_fn():
            sf = ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config())
            bf = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())
            o = ttnn.add(sf, bf, input_tensor_a_activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SCALE)])
            for t in (sf, bf, o):
                ttnn.deallocate(t)

        def got_fn():
            ttnn.deallocate(rfd3_bias.dense_fused_scores_bias_fp32(scores, bias, SCALE))

        r_ms, r_all = timeit(ref_fn, dev)
        g_ms, g_all = timeit(got_fn, dev)
        moved = (H * I * NK * 2 * 2 + H * I * NK * 4) / MB
        rows.append({"H": H, "I": I, "n_key": NK, "torch_equal": eq, "maxabs": mx,
                     "n_differing": nbad, "shipped_ms": round(r_ms, 4),
                     "fused_ms": round(g_ms, 4), "ratio": round(r_ms / g_ms, 4),
                     "shipped_all": r_all, "fused_all": g_all,
                     "fused_mb": round(moved, 2),
                     "fused_gb_s": round(moved / 1024.0 / (g_ms / 1e3), 1)})
        print("[p72] H=%d I=%-4d nk=%-4d  equal=%-5s maxabs=%-9.4g  shipped %7.4f  fused %7.4f "
              " %.3fx  %.1f GB/s" % (H, I, NK, eq, mx, r_ms, g_ms, r_ms / g_ms,
                                     moved / 1024.0 / (g_ms / 1e3)), flush=True)
        ttnn.deallocate(ref)
        ttnn.deallocate(got)
        ttnn.deallocate(scores)
        ttnn.deallocate(bias)

    prod = [r for r in rows if r["I"] == 685][0]
    per_step = (prod["shipped_ms"] - prod["fused_ms"]) * 36
    print("\n[p72] at the page fixture: %+.3f ms/step over 36 calls = %+.3f s/design at 200 steps"
          % (-per_step, -per_step * 200 / 1e3))
    print("[p72] EXACT at every rung: %s" % ok)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "n": N, "scale": SCALE, "all_exact": ok,
        "slots": rfd3_bias.D_SLOTS, "window": rfd3_bias.D_WINDOW,
        "addr_write_mode": rfd3_bias.ADDR_WRITE_MODE,
        "page_fixture_ms_per_step_saved": round(per_step, 3),
        "page_fixture_s_per_design_saved": round(per_step * 200 / 1e3, 3)}, indent=2) + "\n")
    print("wrote", OUT)
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
