"""ESMC's LM FFN gate on Wormhole: today's fc1_a + fc1_b minimal_matmuls and the legacy silu-multiply against
mm_gate.gated (one minimal_matmul over the interleaved weight, the gate in its epilogue), and the stock
minimal_matmul over that same weight without the gate (plain2n, a control for the wider matmul's own cost). Per shape and mode: error of
each against float64 silu(x W1) * (x W2), and median device ms.

    python3 perf/spd_gap_whlm/gate_mm_bench.py --out gate_mm.jsonl
"""
import argparse
import json

import torch
import ttnn

from qkfuse_bench import timed

SHAPES = {"300m": (7, 960, 2560), "600m": (7, 1152, 3072)}   # batch (x 544 rows), d_model, d_ff
S = 544


def err(got, ref):
    e = got - ref
    return {"rel_l2": float(e.norm() / ref.norm()), "max_abs": float(e.abs().max()),
            "pcc": float(torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--modes", default="normal,fast")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio import esmc, mm_gate
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    grid = (int(g.x), int(g.y))
    out = open(args.out, "w")
    for name, (B, d, ff) in SHAPES.items():
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(B, S, d, generator=gen).to(torch.bfloat16)
        w1 = (torch.randn(d, ff, generator=gen) * d ** -0.5).to(torch.bfloat16)
        w2 = (torch.randn(d, ff, generator=gen) * d ** -0.5).to(torch.bfloat16)
        for mode in args.modes.split(","):
            fast = mode == "fast"
            wdt = ttnn.bfloat8_b if fast else ttnn.bfloat16
            ck = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2 if fast else ttnn.MathFidelity.HiFi4,
                math_approx_mode=False, fp32_dest_acc_en=not fast, packer_l1_acc=True)
            xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            wd1, wd2 = (ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt) for w in (w1, w2))
            mmc = esmc._mm_config(xd, wd1)["config"]
            cfg = (mmc.M_block_size, mmc.K_block_size, mmc.N_block_size, mmc.subblock_h, mmc.subblock_w)
            if fast:
                cfg = cfg[:4] + (4,)
            mmc = ttnn.MinimalMatmulConfig(M_block_size=cfg[0], K_block_size=cfg[1], N_block_size=cfg[2],
                                           subblock_h=cfg[3], subblock_w=cfg[4],
                                           compute_with_storage_grid_size=ttnn.CoreCoord(*grid))
            w12 = mm_gate.interleave(w1, w2, cfg[2] // 2 * 32)
            wd12 = ttnn.from_torch(w12, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
            # float64 reference on the weights as the device holds them (block-fp8 rounds them in fast mode)
            w1h, w2h = (ttnn.to_torch(w).double() for w in (wd1, wd2))
            ref = torch.nn.functional.silu(x.double() @ w1h) * (x.double() @ w2h)

            def today():
                h1 = ttnn.experimental.minimal_matmul(xd, wd1, compute_kernel_config=ck, config=mmc,
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
                h2 = ttnn.experimental.minimal_matmul(xd, wd2, compute_kernel_config=ck, config=mmc,
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
                o = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], use_legacy=True)
                ttnn.deallocate(h1)
                ttnn.deallocate(h2)
                return o

            def fused():
                return mm_gate.gated(xd, wd12, cfg, ck, grid)

            ck_f32 = T.silu_ckc(ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ck.math_fidelity, math_approx_mode=True,
                fp32_dest_acc_en=ck.fp32_dest_acc_en, packer_l1_acc=True))

            def fused_f32():   # the epilogue's silu on the silu_f32 overlay (math_approx_mode selects it)
                return mm_gate.gated(xd, wd12, cfg, ck_f32, grid)

            def probe1():   # epilogue without the silu (wrong output, timing only)
                return mm_gate.gated(xd, wd12, cfg, ck, grid, probe=1)

            def probe2():   # epilogue as copy + pack only
                return mm_gate.gated(xd, wd12, cfg, ck, grid, probe=2)

            def plain2n():   # control: the same [W1|W2] matmul, no epilogue, full-width output
                return ttnn.experimental.minimal_matmul(xd, wd12, compute_kernel_config=ck, config=mmc,
                                                        memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

            rec = {"shape": name, "mode": mode, "cfg": list(cfg), "grid": list(grid),
                   "plan": mm_gate.plan(xd, 2 * ff, cfg, grid)}
            for arm, fn in (("today", today), ("fused", fused), ("fused_f32", fused_f32)):
                try:
                    o = fn()
                    rec[arm] = err(ttnn.to_torch(o).double().reshape(ref.shape), ref)
                    ttnn.deallocate(o)
                except Exception as exc:  # noqa: BLE001
                    rec[arm] = {"error": str(exc)[:400]}
            rec["plain2n"] = {}
            rec["probe1"] = rec["probe2"] = {}
            arms = (("today", today), ("fused", fused), ("fused_f32", fused_f32), ("probe1", probe1),
                    ("probe2", probe2), ("plain2n", plain2n))
            ms = {a: [] for a, _ in arms}
            for _ in range(args.repeat):
                for arm, fn in arms:
                    if "error" not in rec[arm]:
                        ms[arm].append(timed(lambda fn=fn: ttnn.deallocate(fn()), dev, 1, args.inner))
            for arm, ts in ms.items():
                if ts:
                    rec[arm + "_ms"] = sorted(ts)[len(ts) // 2]
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
            for t in (xd, wd1, wd2, wd12):
                ttnn.deallocate(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
