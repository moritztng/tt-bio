"""Device time of one Boltz-2 token DiT transition block (512 x 1 sample), DRAM-interleaved vs L1 block-sharded.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/ditshard.py --out ditshard.jsonl

cen14 (WH --fast, 512 x 1): the token DiT spends 5.79 s of device time in 151,200 calls, 86,400 of them under 40 us,
and every [512, 768] activation goes to DRAM and back between ops. Per-op tuning is exhausted (fidelity, program
config, bfp8 weights, dit_l1's L1-interleaved buffers), so this asks what the chain costs if the activation and its
intermediates stay block-sharded in L1 from the adaLN to the residual add. The transition is the part of the layer
with no SDPA in it: LN -> modulate -> swish / gates / a_to_b linears -> SiLU gate -> b_to_a -> gated residual.

Arms: `dram` (what ships: DRAM interleaved), `l1` (L1 interleaved, dit_l1's layout), `shard` (block-sharded on an
8 x 8 grid, matmul programs that read and write shards), `shard_io` (shard, plus the conversion in from and out
to DRAM on every call). Residual fp32, linear operands bf16 with fp32 accumulation as under dit_mm16; the
modulation and gated residual are multiply + add in every arm. Values are random; each arm's output is compared
with `dram` (rel_rms) so a layout that changes the math shows up. Device time is a captured trace replayed back to back (op_probe.dev_time).
"""
import argparse, json, time, traceback

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import dev_time, opened_nodes  # noqa: E402

N, D, H = 512, 768, 1536
GY, GX = 8, 8
TILE = 32


def block(shape, grid=(GY, GX)):
    return ttnn.create_sharded_memory_config(shape=shape, core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]),
                                             strategy=ttnn.ShardStrategy.BLOCK,
                                             orientation=ttnn.ShardOrientation.ROW_MAJOR)


def mm_prog(k, n, act=None):
    """2D multicast program for a block-sharded [N, k] x [k, n] -> block-sharded [N, n] on the GY x GX grid."""
    pm, pk, pn = N // TILE // GY, k // TILE // GX, n // TILE // GX
    sw = max(w for w in (1, 2, 3, 4) if pn % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(GX, GY), in0_block_w=pk, out_subblock_h=1, out_subblock_w=sw,
        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = T.get_device()
    nodes = opened_nodes()
    g = dev.compute_with_storage_grid_size()
    assert g.x >= GX and g.y >= GY, f"grid {g.x}x{g.y} < {GX}x{GY}"
    torch.manual_seed(0)
    up = lambda t, dt=ttnn.bfloat16, mc=None: ttnn.from_torch(  # noqa: E731
        t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt, memory_config=mc or ttnn.DRAM_MEMORY_CONFIG)
    host = dict(a=torch.randn(1, 1, N, D), sc=torch.randn(1, 1, N, D), sh=torch.randn(1, 1, N, D) * .1,
                so=torch.rand(1, 1, N, D))
    wts = dict(sw=torch.randn(D, H) * .03, gt=torch.randn(D, H) * .03, ab=torch.randn(D, H) * .03,
               ba=torch.randn(H, D) * .03)
    W = {k: up(v) for k, v in wts.items()}
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    sd, sh = block((1, 1, N, D)), block((1, 1, N, H))

    def chain(x, mc_d, mc_h, progs, io=False):
        if io:      # inputs arrive in DRAM, the output leaves to DRAM: the per-call cost in a model whose SDPA stays interleaved
            o = chain([ttnn.to_memory_config(t, mc_d) for t in x], mc_d, mc_h, progs)
            out = ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            return out
        a, sc, shf, so = x
        y = ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=ckc, memory_config=mc_d,
                            **({"program_config": progs["ln"]} if progs else {}))
        m = ttnn.add_(ttnn.multiply_(y, sc), shf)
        mb = ttnn.typecast(m, ttnn.bfloat16, memory_config=mc_d)
        ttnn.deallocate(m)
        lin = lambda t, w, k, n: ttnn.linear(  # noqa: E731
            t, w, compute_kernel_config=ckc, memory_config=mc_h if n == H else mc_d, dtype=ttnn.bfloat16,
            **({"program_config": progs[(k, n)]} if progs else {"core_grid": T.CORE_GRID_MAIN}))
        sw_, gt_, ab_ = lin(mb, W["sw"], D, H), lin(mb, W["gt"], D, H), lin(mb, W["ab"], D, H)
        ttnn.deallocate(mb)
        gsw = ttnn.multiply_(gt_, sw_, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(sw_)
        b = ttnn.multiply_(gsw, ab_)
        ttnn.deallocate(ab_)
        ba = lin(b, W["ba"], H, D)
        ttnn.deallocate(b)
        g_ = ttnn.multiply(so, ba, memory_config=mc_d, dtype=ttnn.float32)
        ttnn.deallocate(ba)
        return ttnn.add_(g_, a)

    shard_progs = {
        "ln": ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(GX, GY), subblock_w=D // TILE // GX,
                                                          block_h=N // TILE // GY, block_w=D // TILE // GX, inplace=False),
        (D, H): mm_prog(D, H), (H, D): mm_prog(H, D)}
    arms = {"dram": (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, None),
            "l1": (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, None),
            "shard": (sd, sd, sh, shard_progs), "shard_io": (ttnn.DRAM_MEMORY_CONFIG, sd, sh, shard_progs)}
    rows, ref = [], None
    for arm, (mc_in, mc_d, mc_h, progs) in arms.items():
        io = arm == "shard_io"
        r = dict(arm=arm, arch=T.arch_name(), grid=f"{GY}x{GX}", t=time.time())
        x = None
        try:
            x = (up(host["a"], ttnn.float32, mc_in), up(host["sc"], ttnn.float32, mc_in),
                 up(host["sh"], ttnn.float32, mc_in), up(host["so"], ttnn.float32, mc_in))
            o = chain(x, mc_d, mc_h, progs, io)
            got = ttnn.to_torch(o).float(); ttnn.deallocate(o)
            if ref is None:
                ref = got
            r["rel_rms_vs_dram"] = float((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
            us, spread, clk = dev_time(dev, lambda: chain(x, mc_d, mc_h, progs, io), nodes=nodes)
            r.update(us=us, spread=spread, aiclk=clk)
        except Exception as e:  # noqa: BLE001 -- an arm a layout cannot run is a result, not a crash
            r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            traceback.print_exc()
        finally:
            for t in x or ():
                ttnn.deallocate(t)
        rows.append(r)
        print(json.dumps(r), flush=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
