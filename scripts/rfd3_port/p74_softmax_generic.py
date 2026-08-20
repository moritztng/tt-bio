#!/usr/bin/env python3
"""p74 -- the L5a gates: a transcribed softmax, then the same one packing bf16.

S1 (faithfulness). `tt_bio.softmax_generic` driving the wheel's own kernels with an fp32 output
must reproduce `ttnn.softmax(x, dim=-1)` under `torch.equal` and run within 5 % of it. Until S1
passes, a bf16 result proves nothing -- it could differ because the transcription is wrong rather
than because the packer rounds differently.

S2 (the lever). The same program with `cb_out0` declared bfloat16 must equal
`ttnn.typecast(ttnn.softmax(x, dim=-1), bfloat16)` under `torch.equal`. That is the whole
bit-exactness question: one rounding either way, and this says whether the packer's rounding is
the same rounding `ttnn.typecast` does. A tolerance is not an acceptable substitute -- if S2 fails,
L5a is dead as a bit-exact lever and gets reported as such.

Prize, measured in the same run: shipped `softmax + typecast` against the fused call.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p2 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p2 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p74_softmax_generic.py \
          perf/p74/softmax_generic.json
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
from tt_bio import softmax_generic                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p74/softmax_generic.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MB = 1024.0 * 1024.0

# The four attention sites, at the page fixture. DiT: 36 calls/step at [1,16,685,704].
# Decoder + atom encoder: 9 calls/step at [1,4,6051,6080]. The small rung is the size ladder.
SHAPES = [
    ("dit_128", (1, 16, 128, 128), 36),
    ("dit_512", (1, 16, 512, 512), 36),
    ("dit_685", (1, 16, 685, 704), 36),
    ("atom_6051", (1, 4, 6051, 6080), 9),
]


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
    g = dev.compute_with_storage_grid_size()
    rows = []
    torch.manual_seed(42)

    for name, shape, calls in SHAPES:
        # Scores as attention actually produces them: a bf16 matmul widened to fp32, so the
        # values carry the same exponent spread the real op sees.
        h = (torch.randn(shape, dtype=torch.float32) * 4.0)
        x = ttnn.from_torch(h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

        ref = ttnn.softmax(x, dim=-1)
        ref_bf = ttnn.typecast(ref, ttnn.bfloat16)
        ref_t = ttnn.to_torch(ref)
        ref_bf_t = ttnn.to_torch(ref_bf)

        out32 = ttnn.zeros(list(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        softmax_generic.softmax_into(dev, x, out32, grid=(g.x, g.y))
        got32 = ttnn.to_torch(out32)
        s1_equal = bool(torch.equal(got32, ref_t))
        s1_max = float((got32.float() - ref_t.float()).abs().max())

        out16 = ttnn.zeros(list(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        softmax_generic.softmax_into(dev, x, out16, grid=(g.x, g.y))
        got16 = ttnn.to_torch(out16)
        s2_equal = bool(torch.equal(got16, ref_bf_t))
        s2_max = float((got16.float() - ref_bf_t.float()).abs().max())

        p = softmax_generic.plan(x, out16, (g.x, g.y), True, True)

        t_ship, _ = timeit(lambda: ttnn.typecast(ttnn.softmax(x, dim=-1), ttnn.bfloat16), dev)
        t_soft, _ = timeit(lambda: ttnn.softmax(x, dim=-1), dev)
        t_g32, _ = timeit(lambda: softmax_generic.softmax_into(dev, x, out32, grid=(g.x, g.y)), dev)
        t_g16, _ = timeit(lambda: softmax_generic.softmax_into(dev, x, out16, grid=(g.x, g.y)), dev)

        row = dict(name=name, shape=list(shape), calls_per_step=calls,
                   Wt=p["Wt"], Ht=p["Ht"], block_size=p["block_size"],
                   use_large=p["use_large"], units=p["units"], target_cores=p["target"],
                   s1_equal=s1_equal, s1_maxabs=s1_max,
                   s2_equal=s2_equal, s2_maxabs=s2_max,
                   ms_shipped_softmax_plus_typecast=round(t_ship, 4),
                   ms_native_softmax=round(t_soft, 4),
                   ms_generic_fp32=round(t_g32, 4),
                   ms_generic_bf16=round(t_g16, 4),
                   s1_speed_ratio=round(t_g32 / t_soft, 4),
                   saved_ms_per_call=round(t_ship - t_g16, 4),
                   saved_ms_per_step=round((t_ship - t_g16) * calls, 4))
        rows.append(row)
        print("[p74] %-10s Wt=%-4d large=%-5s S1 %s (%.3g)  S2 %s (%.3g)  "
              "ship %.4f  native %.4f  gen32 %.4f (%.3fx)  gen16 %.4f  -> %+.4f ms/step"
              % (name, p["Wt"], p["use_large"], s1_equal, s1_max, s2_equal, s2_max,
                 t_ship, t_soft, t_g32, t_g32 / t_soft, t_g16,
                 -(t_ship - t_g16) * calls), flush=True)

        for t in (x, ref, ref_bf, out32, out16):
            ttnn.deallocate(t)

    total = sum(r["saved_ms_per_step"] for r in rows if r["name"] in ("dit_685", "atom_6051"))
    summary = dict(rows=rows, grid=[g.x, g.y],
                   s1_all_equal=all(r["s1_equal"] for r in rows),
                   s2_all_equal=all(r["s2_equal"] for r in rows),
                   saved_ms_per_step_page_fixture=round(total, 4),
                   saved_s_per_design_200_steps=round(total * 200 / 1000.0, 4),
                   host="qb2", card=int(os.environ.get("TT_VISIBLE_DEVICES", "1")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print("[p74] S1 all equal: %s   S2 all equal: %s" % (summary["s1_all_equal"],
                                                         summary["s2_all_equal"]))
    print("[p74] page fixture prize %.3f ms/step = %.3f s/design at 200 steps"
          % (total, total * 200 / 1000.0))


if __name__ == "__main__":
    main()
