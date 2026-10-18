"""Program-config sweep of the trimul einsum at OpenDDE's shape on Wormhole.

The trimul's batched matmul is [1, C, S, S] x [1, C, S, S] over the channel batch, C = 384 at
OpenDDE, operands and output in DRAM. At S = 512 (16 tiles) the shipped plan
(`_triangle_mul_program_config`) is the 2D multicast with 2x2 tiles a core on 64 of 72 cores, one
multicast round per channel, in0_block_w 8, subblock 1x1. cs3 reads it at 5.03 ms (20.5 TF, 120 GB/s)
against a 2.4 ms DRAM floor (three pair volumes), 960 calls a fold in both modes.

Arms keep in0_block_w = 8, so K folds in the same order and every arm must be torch.equal to the
shipped plan; an arm that is not is reported and never ranked. Arms that change in0_block_w are
reported separately (`ibw`), as an accuracy-graded lever. The `--fast` operands are bfloat8_b with
no deferred transpose, as `_mm_transpose_deferred` routes them.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/einsum_sweep.py OUT.json [--seq 512] [--c 384]
       [--fid HiFi3] [--fast] [--end]
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--c", type=int, default=384); ap.add_argument("--fid", default="HiFi3")
ap.add_argument("--fast", action="store_true", help="bfloat8_b operands, no deferred transpose")
ap.add_argument("--end", action="store_true", help="the ending node: transpose_a instead of transpose_b")
ap.add_argument("--l1acc", action="store_true"); ap.add_argument("--ncall", type=int, default=4)
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
S, C, St = a.seq, a.c, a.seq // 32
gx, gy = T.COMPUTE_GRID_MAIN
torch.manual_seed(0)
dt = ttnn.bfloat8_b if a.fast else ttnn.bfloat16
up = lambda t: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
A = up(torch.randn(1, C, S, S))
B = up(torch.randn(1, C, S, S) * S ** -0.5)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, a.fid), math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=a.l1acc)
ta, tb = (False, False) if a.fast else ((True, False) if a.end else (False, True))
IBW = T._trimul_in0_block_w(St)


def mcast(ibw, pm, pn, sh, sw):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)


def reuse(ibw, pm, pn, sh, sw):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
        per_core_M=pm, per_core_N=pn)


def run(pc):
    return ttnn.matmul(A, B, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                       program_config=pc, dtype=ttnn.bfloat16, transpose_a=ta, transpose_b=tb)


SHIP = T._triangle_mul_program_config(St, False, (1, 1), False)
arms = {"ship": SHIP}
pm, pn = -(-St // gy), -(-St // gx)
for sh, sw in ((1, 2), (2, 1), (2, 2), (1, 4), (4, 1)):
    if pm % sh == 0 and pn % sw == 0:
        arms[f"mcast {pm}x{pn} sb{sh}x{sw}"] = mcast(IBW, pm, pn, sh, sw)
# whole-K block: one multicast round per channel instead of two (reorders K: not bit-exact)
for sh, sw in ((1, 1), (2, 2)):
    arms[f"ibw mcast ibw{St} sb{sh}x{sw}"] = mcast(St, pm, pn, sh, sw)
# per-core channel blocks without multicast: each core reads its own rows/cols from DRAM
for rm, rn in ((16, 4), (4, 16), (8, 8), (8, 16), (16, 8), (16, 16)):
    if St % rm or St % rn:
        continue
    for sh, sw in ((1, 1), (2, 2), (1, 4), (4, 1)):
        if rm % sh == 0 and rn % sw == 0:
            arms[f"reuse {rm}x{rn} sb{sh}x{sw}"] = reuse(IBW, rm, rn, sh, sw)
            for ib in (2, 4):
                if ib != IBW and St % ib == 0 and (sh, sw) == (2, 2):
                    arms[f"ibw reuse {rm}x{rn} ibw{ib} sb{sh}x{sw}"] = reuse(ib, rm, rn, sh, sw)

ref = ttnn.to_torch(run(SHIP))
res = {"seq": S, "c": C, "fid": a.fid, "fast": a.fast, "end": a.end, "l1acc": a.l1acc, "ibw": IBW,
       "grid": [gx, gy], "card": os.environ["TT_VISIBLE_DEVICES"], "arms": {}}
live = []
for name, pc in arms.items():
    r = res["arms"][name] = {}
    try:
        o = run(pc)
        got = ttnn.to_torch(o)
        o.deallocate()
        r["equal"] = bool(torch.equal(got, ref))
        d = (got.double() - ref.double())
        r["rel_rms"] = float(d.pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt())
        live.append(name)
    except Exception as e:  # an L1 refusal or a config the factory asserts on
        r["error"] = str(e).splitlines()[0][:200]
    print(name, r, flush=True)

for rep in range(a.reps):
    for name in live:
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.ncall):
            run(arms[name]).deallocate()
        ttnn.synchronize_device(dev)
        res["arms"][name].setdefault("ms", []).append((time.perf_counter() - t0) * 1e3 / a.ncall)
for name in live:
    r = res["arms"][name]
    r["ms_med"] = round(statistics.median(r["ms"]), 3)
    print(f"{r['ms_med']:8.3f} ms  equal={r['equal']}  rel_rms={r['rel_rms']:.2e}  {name}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
