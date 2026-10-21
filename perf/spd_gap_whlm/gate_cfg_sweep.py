"""Block-config sweep for mm_gate.gated (PACK-thread epilogue) at ESMC's LM FFN page shapes: every (M_block, N_block,
subblock_w) that `mm_gate.plan` accepts, against today's two minimal_matmuls + silu-multiply. Median device ms per
config, and the error of the best one against float64.

    python3 perf/spd_gap_whlm/gate_cfg_sweep.py --out sweep.jsonl
"""
import argparse
import json

import torch
import ttnn

from qkfuse_bench import timed
from gate_mm_bench import SHAPES, S, err

SHAPES = {**SHAPES, "6b": (1, 2560, 6912)}   # the 6B's "today" is its LIN6B linears (544 rows, below minimal_matmul's)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--shapes", default="300m,600m")
    ap.add_argument("--fidelity", default=None, help="one MathFidelity for every arm (the 300M/600M model runs HiFi3); "
                    "default HiFi2 --fast, HiFi4 normal and for the 6B")
    ap.add_argument("--modes", default="normal,fast")
    ap.add_argument("--full-sync", action="store_true", help="dst_full_sync_en for every arm: dest holds twice the "
                    "subblock tiles (8 fp32 / 16 bf16) but math and pack no longer overlap on dest halves")
    ap.add_argument("--kb", default=None, help="comma list of K block sizes in tiles (default: the model's, 2 and 4)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import metal_overlay
    metal_overlay.enable(("silu_f32",))
    from tt_bio import tenstorrent as T
    from tt_bio import esmc, mm_gate
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    grid = (int(g.x), int(g.y))
    out = open(args.out, "w")
    for name in args.shapes.split(","):
        B, d, ff = SHAPES[name]
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(B, S, d, generator=gen).to(torch.bfloat16)
        w1 = (torch.randn(d, ff, generator=gen) * d ** -0.5).to(torch.bfloat16)
        w2 = (torch.randn(d, ff, generator=gen) * d ** -0.5).to(torch.bfloat16)
        for mode in args.modes.split(","):
            fast = mode == "fast"
            wdt = ttnn.bfloat8_b if fast else ttnn.bfloat16
            ck = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, args.fidelity) if args.fidelity else
                ttnn.MathFidelity.HiFi2 if fast and name != "6b" else ttnn.MathFidelity.HiFi4,
                math_approx_mode=False, fp32_dest_acc_en=not fast or name == "6b", packer_l1_acc=True,   # the 6B keeps HiFi4
                dst_full_sync_en=args.full_sync)
            xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            wd1, wd2 = (ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt) for w in (w1, w2))
            base = esmc._mm_config(xd, wd1)["config"]   # 6b: only its K block seeds the sweep
            mmc = ttnn.MinimalMatmulConfig(M_block_size=base.M_block_size, K_block_size=base.K_block_size,
                                           N_block_size=base.N_block_size, subblock_h=base.subblock_h,
                                           subblock_w=4 if fast else base.subblock_w,
                                           compute_with_storage_grid_size=ttnn.CoreCoord(*grid))

            def today_6b():
                h1, h2 = (ttnn.linear(xd, w, program_config=esmc._lin6b_program(xd, w), compute_kernel_config=ck,
                                      dtype=ttnn.bfloat16) for w in (wd1, wd2))
                o = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], use_legacy=True)
                ttnn.deallocate(h1)
                ttnn.deallocate(h2)
                return o

            def today():
                if name == "6b":
                    return today_6b()
                h1 = ttnn.experimental.minimal_matmul(xd, wd1, compute_kernel_config=ck, config=mmc,
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
                h2 = ttnn.experimental.minimal_matmul(xd, wd2, compute_kernel_config=ck, config=mmc,
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
                o = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], use_legacy=True)
                ttnn.deallocate(h1)
                ttnn.deallocate(h2)
                return o

            ts = sorted(timed(lambda: ttnn.deallocate(today()), dev, 1, args.inner) for _ in range(args.repeat))
            rec = {"shape": name, "mode": mode, "fidelity": str(ck.math_fidelity), "full_sync": args.full_sync, "today_ms": ts[len(ts) // 2], "configs": []}
            w1h, w2h = (ttnn.to_torch(w).double() for w in (wd1, wd2))
            ref = torch.nn.functional.silu(x.double() @ w1h) * (x.double() @ w2h)
            dmax = (4 if not fast or name == "6b" else 8) * (2 if args.full_sync else 1)   # dest tiles a subblock gets
            for kb in sorted({int(k) for k in args.kb.split(",")} if args.kb else {base.K_block_size, 2, 4}):
                if (d // 32) % kb:
                    continue
                for nb in (4, 6, 8, 10, 12):
                    wd12 = None
                    for mb in ((1, 2, 4, 8) if name == "6b" else (8, 16)):
                        for sw in (2, 3, 4, 5, 6, 8):
                            for sh in (1, 2, 4):
                                if nb % sw or mb % sh or sh * sw > dmax:
                                    continue
                                cfg = (mb, kb, nb, sh, sw)
                                if mm_gate.plan(xd, 2 * ff, cfg, grid) is not None:
                                    continue
                                if wd12 is None:
                                    wd12 = ttnn.from_torch(mm_gate.interleave(w1, w2, nb // 2 * 32),
                                                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
                                fn = lambda cfg=cfg, w=wd12: mm_gate.gated(xd, w, cfg, ck, grid, pack_sfpu=True)
                                try:
                                    o = fn()
                                    e = err(ttnn.to_torch(o).double().reshape(ref.shape), ref)
                                    ttnn.deallocate(o)
                                    t = sorted(timed(lambda: ttnn.deallocate(fn()), dev, 1, args.inner)
                                               for _ in range(args.repeat))
                                    rec["configs"].append({"cfg": cfg, "ms": t[len(t) // 2], "rel_l2": e["rel_l2"]})
                                except Exception as exc:  # noqa: BLE001
                                    rec["configs"].append({"cfg": cfg, "error": str(exc)[:200]})
                    if wd12 is not None:
                        ttnn.deallocate(wd12)
            ok = [c for c in rec["configs"] if "ms" in c]
            rec["best"] = min(ok, key=lambda c: c["ms"]) if ok else None
            print(json.dumps({k: v for k, v in rec.items() if k != "configs"}), flush=True)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            for t in (xd, wd1, wd2):
                ttnn.deallocate(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
