"""Block sweep of OpenDDE's fused triangle-attention projection on Wormhole.

`triatt_qkv.qkvgb_heads` projects the normed pair [512, 512, 384] through [384, 1568] (q, k, v, gate,
pair bias) in one head-major `generic_minimal_matmul`. Its block entry is derived from the (12, 36)
literal, (M 4, K 12, N 1, subblock 2x1), and that was swept on Blackhole only (perf/odde4x). In the
cs1 census it is 9.3 ms a call, 30.7 TF, 101 GB/s: 11.3 s of the fold.

K_block stays 12, the whole contraction, so every arm folds K in the same order as today and must be
torch.equal to it; an arm that is not is reported and never ranked. Arms time NCALL calls between
two syncs, REPS rounds, interleaved.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/qkvgb_sweep.py OUT.json [--seq 512] [--heads 12] [--fid HiFi4]
"""
import argparse, itertools, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--heads", type=int, default=12, help="12 OpenDDE, 8 Protenix-v2, 4 Boltz-2 (head_dim 32)")
ap.add_argument("--fid", default="HiFi4"); ap.add_argument("--ncall", type=int, default=6)
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--rule", action="store_true", help="only ship vs the shipped Wormhole rule (16, KT, 1, 4, 1)")
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import mm_generic as G, triatt_qkv as Q

dev = T.get_device()
S, H, D = a.seq, a.heads, 32
C = H * D
KT = C // 32
NW = 4 * C + 32
torch.manual_seed(0)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
x = up(torch.randn(S, S, C))
w = up(torch.randn(C, NW) * C ** -0.5)
ckc = (getattr(ttnn.MathFidelity, a.fid), False, True, False)
grid = tuple(T.COMPUTE_GRID_MAIN)
pad = [int(d) for d in x.padded_shape]


def run(blk):
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([S, H, S, D]),
                                           ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(4)]
    outs.append(ttnn.allocate_tensor_on_device(ttnn.Shape([S, S, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                               dev, ttnn.DRAM_MEMORY_CONFIG))
    G.generic_minimal_matmul(dev, x, w, outs, (blk, grid), ckc, {"HEAD_MAJOR_MT": pad[-2] // 32},
                             Q.KERNEL_DIR, n_widths=[C // 32] * 4 + [1], guard="tri")
    return outs


SHIP = T._mm_block_at(KT, 4 * KT + 1)
ref = [ttnn.to_torch(o) for o in run(SHIP)]
arms = [SHIP]
RULE = (16, KT, 1, 4, 1)
for M, N, (sh, sw) in itertools.product((2, 4, 8, 16), (1, 2, 3, 6), ((2, 1), (4, 1), (1, 2), (2, 2), (1, 3), (1, 1))):
    blk = (M, KT, N, sh, sw)
    if M % sh or N % sw or blk == SHIP or sh * sw > 4 or (a.rule and blk != RULE):
        continue
    arms.append(blk)
res = {"seq": S, "heads": H, "fid": a.fid, "grid": list(grid), "ship": list(SHIP), "arch": str(dev.arch()), "arms": {}}
ok = []
for blk in arms:
    r = res["arms"][str(blk)] = {}
    try:
        outs = run(blk)
        r["equal"] = all(torch.equal(ttnn.to_torch(o), e) for o, e in zip(outs, ref))
        for o in outs:
            o.deallocate()
    except Exception as e:  # an L1 refusal or a config the kernel asserts on
        r["error"] = str(e).splitlines()[0][:200]
        continue
    if r["equal"]:
        ok.append(blk)
    print(blk, r, flush=True)
times = {blk: [] for blk in ok}
for _ in range(a.reps):
    for blk in ok:
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(a.ncall):
            for o in run(blk):
                o.deallocate()
        ttnn.synchronize_device(dev)
        times[blk].append((time.perf_counter() - t0) / a.ncall * 1e3)
for blk, t in sorted(times.items(), key=lambda kv: statistics.median(kv[1])):
    res["arms"][str(blk)].update(ms_median=statistics.median(t), ms=t)
    print(f"{str(blk):24s} {statistics.median(t):7.3f} ms{'  <- ship' if blk == SHIP else ''}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
