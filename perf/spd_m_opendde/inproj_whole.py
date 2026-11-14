"""The trimul in-projection as one split call against today's one call per channel chunk.

cs6 (OpenDDE 256 tokens, WH 1000 MHz) prices `_in_proj_matmul` x [S*S,384] @ [384,128] at ~800 us, 12 calls per
trimul (chunk 32 x a, b and both gates), 0.33 of roof: the [S*S,384] input is re-read once per chunk and the
4-tile output pads to the 8-core grid axis. `mm_dualnoc.in_proj(split=...)` writes several destination buffers
from one program, so all 12 chunks' projections can come from one read of the input: [384,1536] split 12 x 128.
Each output tile is its own contraction, so every arm is checked torch.equal against the per-chunk arm.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/inproj_whole.py OUT.json [--sizes 256 288 320] [--chunks 12]
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--sizes", nargs="+", type=int, default=[256, 288, 320, 384])
ap.add_argument("--c", type=int, default=384)
ap.add_argument("--chunk-cols", type=int, default=128)
ap.add_argument("--chunks", type=int, default=12)
ap.add_argument("--groups", nargs="+", type=int, default=[1, 2, 3, 4, 6, 12], help="chunks per split call")
ap.add_argument("--fid", default="HiFi3")
ap.add_argument("--ncall", type=int, default=3); ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import mm_dualnoc as DN

dev = T.get_device()
DRAM = ttnn.DRAM_MEMORY_CONFIG
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, a.fid), math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn):
    ts = []
    for _ in range(a.reps):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(a.ncall):
            outs = fn()
            for o in outs:
                o.deallocate()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / a.ncall * 1e3)
    return statistics.median(ts), ts


res = {"arch": str(dev.arch()), "fid": a.fid, "rows": []}
cc, n = a.chunk_cols, a.chunks
for S in a.sizes:
    torch.manual_seed(S)
    x = up(torch.randn(S * S, a.c).bfloat16())
    wt = (torch.randn(a.c, cc * n) / a.c ** 0.5).bfloat16()
    w_chunks = [up(wt[:, i * cc:(i + 1) * cc].contiguous()) for i in range(n)]
    DN.STATS[0] = 0
    ref = [T._in_proj_matmul(x, w, ckc, DRAM) for w in w_chunks]
    ref_dn = DN.STATS[0]
    ref_t = [ttnn.to_torch(o) for o in ref]
    for o in ref:
        o.deallocate()
    base, base_ts = timed(lambda: [T._in_proj_matmul(x, w, ckc, DRAM) for w in w_chunks])
    res["rows"].append({"S": S, "group": 1, "ms_per_trimul": base, "reps": base_ts, "dualnoc_calls": ref_dn, "equal": True})
    print(S, "per-chunk", round(base, 3), "ms / 12 chunks", "dualnoc", ref_dn, flush=True)
    for g in a.groups:
        if g == 1 or n % g:
            continue
        w_groups = [up(wt[:, j * g * cc:(j + 1) * g * cc].contiguous()) for j in range(n // g)]
        run = lambda: [o for w in w_groups for o in (DN.in_proj(x, w, ckc, ttnn.bfloat16, DRAM, split=[cc] * g) or [])]
        outs = run()
        row = {"S": S, "group": g}
        if len(outs) != n:
            row["declined"] = True
        else:
            row["equal"] = all(torch.equal(ttnn.to_torch(o), r) for o, r in zip(outs, ref_t))
            for o in outs:
                o.deallocate()
            row["ms_per_trimul"], row["reps"] = timed(run)
            row["saving_ms"] = base - row["ms_per_trimul"]
        print(S, "group", g, row, flush=True)
        res["rows"].append(row)
        for w in w_groups:
            w.deallocate()
    x.deallocate()
    json.dump(res, open(a.out, "w"), indent=1)
json.dump(res, open(a.out, "w"), indent=1)
print("DONE")
