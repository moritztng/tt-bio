"""Where a SaProt projection's device time goes, per RISC: the model's call and the shipped pick for each projection at
the page shape (7 x 544 rows), plus a K ladder of fc1's call at fixed M, N, under the device profiler.

fc1 runs at ~half fc2's rate for the same FLOPs on both arches (WH --fast 1.24 vs 0.68 ms, BH 0.56 vs 0.29), and no
program knob moved it (lin_sweep ls1/ls33/ls38/ls45). This reads, per program, the kernel duration of each RISC and the
compute cluster's stall accumulators (TT_METAL_PROFILER_SUM: CB WAIT FRONT on the unpacker, CB RESERVE BACK on the
packer), each divided by the program's own core count, so the per-output-tile cost lands on a thread.

Needs the Tracy tt-metal build and TT_METAL_DEVICE_PROFILER=1 (perf/spd_m_saprot/prof.sh sets the environment).

    python perf/spd_m_saprot/mm_prof.py --model saprot-650m [--fast] --out prof.jsonl
"""
import argparse
import json
import math

import torch
import ttnn

MODELS = {"saprot-35m": (480, 1920), "saprot-650m": (1280, 5120), "saprot-1.3b": (1280, 5120)}
KEYS = {"k": "DEVICE KERNEL DURATION [ns]", "br": "DEVICE BRISC KERNEL DURATION [ns]",
        "nc": "DEVICE NCRISC KERNEL DURATION [ns]", "t0": "DEVICE TRISC0 KERNEL DURATION [ns]",
        "t1": "DEVICE TRISC1 KERNEL DURATION [ns]", "t2": "DEVICE TRISC2 KERNEL DURATION [ns]"}
SUMS = {"wait": "DEVICE COMPUTE CB WAIT FRONT [ns]", "resv": "DEVICE COMPUTE CB RESERVE BACK [ns]"}


def _sub8(h, w):
    """The largest out subblock (h, w) dividing an h x w block within 8 dest tiles, widest first."""
    return max(((sh, sw) for sh in range(1, 9) for sw in range(1, 9) if sh * sw <= 8 and h % sh == 0 and w % sw == 0),
               key=lambda t: (t[0] * t[1], t[1]))


def fullsync_arm(prog, xd, wd, bd, ck):
    """The picked program with dst_full_sync_en and a subblock of up to 8 tiles (fp32 dest at full sync holds 8)."""
    ckfs = type(ck)(math_fidelity=ck.math_fidelity, math_approx_mode=ck.math_approx_mode,
                    fp32_dest_acc_en=ck.fp32_dest_acc_en, packer_l1_acc=ck.packer_l1_acc, dst_full_sync_en=True)
    kind, c = prog
    if kind == "mm":
        sh, sw = _sub8(c.M_block_size, c.N_block_size)
        cfg = ttnn.MinimalMatmulConfig(M_block_size=c.M_block_size, K_block_size=c.K_block_size,
                                       N_block_size=c.N_block_size, subblock_h=sh, subblock_w=sw,
                                       compute_with_storage_grid_size=c.compute_with_storage_grid_size)
        return lambda: ttnn.experimental.minimal_matmul(xd, wd, bias_tensor=bd, compute_kernel_config=ckfs,
                                                         dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                         config=cfg)
    sh, sw = _sub8(c.out_block_h, c.out_block_w)
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=c.compute_with_storage_grid_size, in0_block_w=c.in0_block_w, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=c.out_block_h, out_block_w=c.out_block_w, per_core_M=c.per_core_M,
        per_core_N=c.per_core_N, transpose_mcast=c.transpose_mcast, fused_activation=c.fused_activation, fuse_batch=True)
    return lambda: ttnn.linear(xd, wd, bias=bd, program_config=cfg, compute_kernel_config=ckfs, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m", choices=sorted(MODELS))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--rows", type=int, default=7 * 544)
    ap.add_argument("--shapes", default="qkv,out,fc1,fc2")
    ap.add_argument("--kladder", default="320,640,1280,2560", help="fc1's call at these K (N, M fixed); empty = none")
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--cfg2d", default="", help="extra fc1 arms (approx gelu), ';'-separated tm,kb,sh,sw,oh,ow on the "
                    "2D mcast program at the model grid, per_core_M/N from the grid")
    ap.add_argument("--l1x", action="store_true", help="the --fast page's L1 state: x in L1 plus a resident [rows, d] "
                    "bf16 L1 tensor (the residual under _SAPROT_X_L1)")
    ap.add_argument("--fullsync", action="store_true", help="per shape, the pick again with the full-sync dest (8 fp32 "
                    "tiles instead of 4) and the largest subblock that fits it: arm pickfs, plus max |pick - pickfs|")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T, ops
    from tt_bio import saprot as S
    T.set_fast_mode(a.fast)
    dev = T.get_device()
    cfg, _ = S.CONFIGS[a.model]
    ck = S.Saprot(**cfg).compute_kernel_config
    d, ff = MODELS[a.model]
    dh = d if (d // cfg["n_heads"]) % 64 == 0 else cfg["n_heads"] * 32
    shapes = {"qkv": (d, 3 * dh, None), "out": (dh, d, None), "fc1": (d, ff, "gelu"), "fc2": (ff, d, None),
              "fc1n": (d, ff, None),  # fc1 without its fused GELU: the activation's device cost
              "fc1a": (d, ff, "gelu_approx")}  # approx gelu: calculate_gelu_f32 under the gelu_f32 overlay
    for k in filter(None, a.kladder.split(",")):
        shapes[f"fc1k{k}"] = (int(k), ff, "gelu")
    wdt = ttnn.bfloat8_b if a.fast else ttnn.bfloat16
    out = open(a.out, "a")
    gen = torch.Generator().manual_seed(0)
    B, L = 7, a.rows // 7

    def flush():
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        return [p for progs in ttnn.get_latest_programs_perf_data().values() for p in progs]

    flush()
    for name in a.shapes.split(",") + [s for s in shapes if s.startswith("fc1k")]:
        K, N, act = shapes[name]
        x = torch.randn(B, L, K, generator=gen).to(torch.bfloat16)
        w = (torch.randn(K, N, generator=gen) / math.sqrt(K)).to(torch.bfloat16)
        b = (torch.randn(N, generator=gen) * 0.1).to(torch.bfloat16)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
        mc = ttnn.L1_MEMORY_CONFIG if a.l1x else ttnn.DRAM_MEMORY_CONFIG
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
        if a.l1x:
            resid = ttnn.from_torch(torch.zeros(B, L, d, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                                    dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        bd = ttnn.from_torch(b.reshape(1, N), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        kw = {"activation": act} if act else {}
        lin = S._SaprotLin.__new__(S._SaprotLin)
        lin.compute_kernel_config, lin.norm_weight = ck, torch.empty(d)
        mck = lin._ck(wd)
        arms = {"model": lambda: ops.linear(xd, wd, bias=bd, compute_kernel_config=mck, dtype=ttnn.bfloat16,
                                            core_grid=T.CORE_GRID_MAIN, **kw)}
        hifi2 = mck.math_fidelity == ttnn.MathFidelity.HiFi2
        prog = None if name.startswith("fc1k") else S._saprot_prog(xd, wd, act, hifi2)
        if prog is not None:
            arms["pick"] = lambda: lin._lin(xd, wd, bias=bd, **kw)
        if prog is not None and a.fullsync:
            arms["pickfs"] = fullsync_arm(prog, xd, wd, bd, mck)
        if name == "fc1a":
            g = T.CORE_GRID_MAIN
            for spec in filter(None, a.cfg2d.split(";")):
                tm, kb, sh, sw, oh, ow = (int(v) for v in spec.split(","))
                rows_ax, cols_ax = (g.x, g.y) if tm else (g.y, g.x)
                pm, pn = -(-(a.rows // 32) // rows_ax), -(-(N // 32) // cols_ax)
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y), in0_block_w=kb, out_subblock_h=sh,
                    out_subblock_w=sw, out_block_h=oh or pm, out_block_w=ow or pn, per_core_M=pm, per_core_N=pn,
                    transpose_mcast=bool(tm), fuse_batch=True,
                    fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0))
                arms[f"2d:{spec}"] = lambda pc=pc: ttnn.linear(xd, wd, bias=bd, program_config=pc, compute_kernel_config=mck,
                                                               dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if "pickfs" in arms:
            try:
                ref, alt = (ttnn.to_torch(arms[k]()).float() for k in ("pick", "pickfs"))
                rec = dict(model=a.model, fast=a.fast, shape=name, arm="pickfs_diff", max_abs=float((ref - alt).abs().max()),
                           ndiff=int((ref != alt).sum()), numel=ref.numel())
            except Exception as e:  # noqa: BLE001
                rec = dict(model=a.model, fast=a.fast, shape=name, arm="pickfs_diff", error=str(e).splitlines()[0][:200])
            print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
        for arm, fn in arms.items():
            try:
                for _ in range(2):
                    ttnn.deallocate(fn())
                flush()
                for _ in range(a.calls):
                    ttnn.deallocate(fn())
                progs = flush()
            except Exception as e:  # noqa: BLE001 -- a refused program is a data point
                rec = dict(model=a.model, fast=a.fast, shape=name, arm=arm, error=str(e).splitlines()[0][:200])
                print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
                continue
            for p in progs:
                r = p.program_analyses_results
                cc = max(1, int(p.core_count))
                rec = dict(model=a.model, fast=a.fast, shape=name, arm=arm, M=a.rows, K=K, N=N, cores=cc,
                           tiles_out=(a.rows // 32) * (N // 32), kt=K // 32,
                           **{s: round(r[k].duration / 1e3, 2) for s, k in KEYS.items() if k in r},
                           **{s: round(r[k].duration / cc / 1e3, 2) for s, k in SUMS.items() if k in r})
                print(json.dumps(rec), flush=True)
                out.write(json.dumps(rec) + "\n"); out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
