"""Validate the CHANNEL shard: does trimul's per-channel contraction split across the pair?

Finding O: TriangleMultiplication already loops over channel chunks and concatenates them once on
the hidden axis before its channel-mixing tail, and `x[i,j,c] = sum_k a[i,k,c] b[j,k,c]` is
independent per channel. So the shard is half the loop per chip plus one all_gather on the hidden
axis -- a loop bound and a collective, not a rewrite.

There is one thing that plan has to get right and a projection cannot check. ttnn's mesh is SPMD:
both chips run the SAME program, so they cannot be made to compute different channels by control
flow. The channels have to differ because the DATA differs -- the per-channel weights are sharded
on the hidden axis and each chip's loop reads its own shard. This builds exactly that and measures
whether it is bit-exact and whether it pays.

REPLICATED  each chip holds all 128 channels and does the whole contraction. Wall clock = one chip.
SHARDED     each chip holds 64 channels, contracts them, then all_gather on the hidden axis.
"""

import json
import os
import statistics
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("CHAN_OUT", "/tmp/b2z2_chan.json")
BURST, REPS = 4, 5
C, L = 128, 512          # production hidden width and 512 aa pair track
CardSetLease().acquire()
import ttnn  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)
repl = ttnn.replicate_tensor_to_mesh_mapper(mesh)
chan = ttnn.shard_tensor_to_mesh_mapper(mesh, 1)     # hidden axis
res = {"channels": C, "pair": L}

try:
    torch.manual_seed(0)
    a = torch.empty(1, C, L, L, dtype=torch.bfloat16).uniform_(-1, 1)
    b = torch.empty(1, C, L, L, dtype=torch.bfloat16).uniform_(-1, 1)
    nb = a.numel() * 2
    log(f"per-tensor {nb/1e6:.1f} MB, hidden {C} channels")

    a_rep = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
    b_rep = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
    a_shd = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=chan)
    b_shd = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=chan)

    full = lambda: ttnn.matmul(a_rep, b_rep, compute_kernel_config=KC)
    def sharded():
        o = ttnn.matmul(a_shd, b_shd, compute_kernel_config=KC)
        g = ttnn.all_gather(o, dim=1)
        ttnn.deallocate(o)
        return g

    # ---- parity: does the channel shard reproduce the unsharded result exactly ----------
    r = ttnn.to_torch(full(), mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, 0))[0:1]
    s = ttnn.to_torch(sharded(), mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, 0))[0:1]
    ok = torch.equal(r.float(), s.float())
    bad = s.clone(); bad[0, 100, 7, 3] = bad[0, 100, 7, 3] + 1.0
    ctrl = not torch.equal(r.float(), bad.float())
    res["bit_exact"] = bool(ok)
    res["negative_control_rejects"] = bool(ctrl)
    res["max_abs_diff"] = float((r.float() - s.float()).abs().max())
    log(f"PARITY bit_exact={ok}  negctrl_rejects={ctrl}  maxdiff={res['max_abs_diff']}")

    def bench(fn):
        v = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            outs = [fn() for _ in range(BURST)]
            ttnn.synchronize_device(mesh)
            v.append((time.perf_counter() - t0) / BURST)
            for o in outs:
                ttnn.deallocate(o)
        return statistics.median(v)

    bench(full); bench(sharded)                       # warm
    t_full, t_shd = bench(full), bench(sharded)
    # bytes the replicated arm moves: a + b + out
    traffic = 3 * nb
    res.update({"replicated_s": t_full, "sharded_with_gather_s": t_shd,
                "speedup": t_full / t_shd,
                "implied_gbps_replicated": traffic / t_full / 1e9,
                "physically_possible": traffic / t_full / 1e9 <= 444.9})
    log(f"REPLICATED (=1 chip)     {t_full*1e3:8.3f} ms   implied {traffic/t_full/1e9:6.1f} GB/s "
        f"{'OK' if res['physically_possible'] else '*** ABOVE DRAM ROOF ***'}")
    log(f"SHARDED + all_gather     {t_shd*1e3:8.3f} ms   SPEEDUP {t_full/t_shd:.3f}x")
except Exception as e:
    import traceback
    res["error"] = f"{type(e).__name__}: {e}"
    log(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    try:
        ttnn.close_mesh_device(mesh)
        log("closed cleanly")
    except Exception as e:
        log(f"close failed: {e}")
json.dump(res, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")
os._exit(0)
