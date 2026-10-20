"""The Boltz-2 token DiT adaLN (512 x 1, fp32 activation): is ttnn's interleaved LayerNorm core-starved, and does the
block-sharded LayerNorm (cross-core row reduction) fix it? Device time and error against a float64 reference.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/adaln_ln.py --out adaln_ln.jsonl

  ln          ttnn.layer_norm on the DRAM-interleaved activation, as AdaLN.__call__ ships (one core per tile row: 16)
  ln_mod      ln, then `sb + ln * sc` through addcmul_at (the shipped Wormhole fast adaLN, s-terms dense here)
  ln_l1_mod   ln_mod with the LN output in L1 interleaved (`adaln_l1`)
  sh<gx>x<gy>      the sharded LayerNorm on an L1 block-sharded copy, interleave -> shard -> LN -> DRAM included
  sh<gx>x<gy>_ln   the sharded LayerNorm alone, input already sharded (the bound if the producer wrote it sharded)
rel = ||out - ref|| / ||ref||, ref in float64 from the same fp32 operands.
"""
import argparse, json, time, traceback

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import dev_time, opened_nodes  # noqa: E402

N, D, EPS = 512, 768, 1e-5


def rel(t, ref):
    x = ttnn.to_torch(t).double().reshape(ref.shape)
    return float((x - ref).norm() / ref.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = T.get_device()
    nodes = opened_nodes()
    torch.manual_seed(0)
    a_h = (torch.randn(N, D) * 2 + 0.5).float().double()
    sc_h, sb_h = torch.randn(N, D).float().double(), torch.randn(N, D).float().double()
    ln_ref = (a_h - a_h.mean(-1, keepdim=True)) / torch.sqrt(a_h.var(-1, unbiased=False, keepdim=True) + EPS)
    mod_ref = sb_h + ln_ref * sc_h
    up = lambda t: ttnn.from_torch(t.reshape(1, 1, N, D).float(), layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                   dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    a, sc, sb = up(a_h), up(sc_h), up(sb_h)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)

    def ln():
        return ttnn.layer_norm(a, epsilon=EPS, compute_kernel_config=ckc)

    def ln_mod():
        y = ln()
        out = T._addcmul_at(sb, y, sc, D)
        ttnn.deallocate(y)
        return out

    def ln_l1_mod():
        y = ttnn.layer_norm(a, epsilon=EPS, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = T._addcmul_at(sb, y, sc, D)
        ttnn.deallocate(y)
        return out

    arms = {"ln": (ln, ln_ref), "ln_mod": (ln_mod, mod_ref), "ln_l1_mod": (ln_l1_mod, mod_ref)}
    sharded = {}
    for gx, gy in ((8, 8), (6, 8), (4, 8), (8, 4)):
        bh, bw = N // 32 // gy, D // 32 // gx
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(cores, [bh * 32, bw * 32], ttnn.ShardOrientation.ROW_MAJOR))
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(gx, gy),
                                                         subblock_w=1, block_h=bh, block_w=bw, inplace=False)

        def full(mc=mc, pc=pc):
            s = ttnn.to_memory_config(a, mc)
            y = ttnn.layer_norm(s, epsilon=EPS, compute_kernel_config=ckc, program_config=pc, memory_config=mc)
            ttnn.deallocate(s)
            out = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(y)
            return out

        def only(mc=mc, pc=pc, key=(gx, gy)):
            if key not in sharded:
                sharded[key] = ttnn.to_memory_config(a, mc)
            return ttnn.layer_norm(sharded[key], epsilon=EPS, compute_kernel_config=ckc, program_config=pc,
                                   memory_config=mc)

        arms[f"sh{gx}x{gy}"] = (full, ln_ref)
        arms[f"sh{gx}x{gy}_ln"] = (only, ln_ref)
    rows = []
    for arm, (fn, ref) in arms.items():
        r = dict(arm=arm, arch=T.arch_name(), t=time.time())
        try:
            y = fn()
            r["rel"] = rel(y, ref)
            ttnn.deallocate(y)
            us, spread, clk = dev_time(dev, fn, nodes=nodes)
            r.update(us=us, spread=spread, aiclk=clk)
        except Exception as e:  # noqa: BLE001 -- an arm that cannot run is a result, not a crash
            r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            traceback.print_exc()
        rows.append(r)
        print(json.dumps(r), flush=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
