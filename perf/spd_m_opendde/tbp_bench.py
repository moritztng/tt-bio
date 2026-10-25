"""Triangle attention's bias to head-major, isolated: the [1, S, S, H] projection output permuted to [1, H, S, S]
(starting node), plus the ending node's second (0, 1, 3, 2) permute. cs10 reads the first at 476 us per call in-fold;
the in-fold census inflated other ops 2x, so this measures the real cost before a fused head-major writer is built.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/tbp_bench.py OUT.json [--seq 640] [--heads 12]
"""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seq", type=int, nargs="+", default=[640, 608, 288])
ap.add_argument("--heads", type=int, default=12); ap.add_argument("--ncall", type=int, default=20); ap.add_argument("--reps", type=int, default=5)
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
res = {"heads": a.heads, "arch": str(dev.arch()), "seq": {}}
for s in a.seq:
    torch.manual_seed(0)
    x = torch.randn(1, s, s, a.heads)
    b = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    hm = ttnn.permute(b, (0, 3, 1, 2))

    def _tr_perm(t):
        u = ttnn.transpose(t, -2, -1)  # [1, S, H, S], in-tile transpose
        o = ttnn.permute(u, (0, 2, 1, 3)); u.deallocate(); return o

    def _rm_perm(t):
        u = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        v = ttnn.permute(u, (0, 3, 1, 2)); u.deallocate()
        o = ttnn.to_layout(v, ttnn.TILE_LAYOUT); v.deallocate(); return o

    cases = {"start_0312": lambda: ttnn.permute(b, (0, 3, 1, 2)),
             "end_0132_after": lambda: ttnn.permute(hm, (0, 1, 3, 2)),
             "end_0321_once": lambda: ttnn.permute(b, (0, 3, 2, 1)),
             "start_tr_then_0213": lambda: _tr_perm(b),
             "start_rm": lambda: _rm_perm(b)}
    r = {}
    for name, f in cases.items():
        o = f(); got = ttnn.to_torch(o).float(); o.deallocate()
        want = {"start_0312": x.permute(0, 3, 1, 2), "end_0132_after": x.permute(0, 3, 2, 1),
                "end_0321_once": x.permute(0, 3, 2, 1),
                "start_tr_then_0213": x.permute(0, 3, 1, 2), "start_rm": x.permute(0, 3, 1, 2)}[name]
        ms = []
        for _ in range(a.reps):
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            for _ in range(a.ncall):
                f().deallocate()
            ttnn.synchronize_device(dev)
            ms.append((time.perf_counter() - t0) / a.ncall * 1e3)
        r[name] = {"ms_min": min(ms), "ms": ms, "equal": bool(torch.equal(got, want.to(torch.bfloat16).float()))}
    # bytes moved: read padded H->32 tiles [S, S, 32] + write [H, S, S], bf16
    r["bytes_rw"] = 2 * (s * s * 32 + a.heads * s * s)
    res["seq"][s] = r
    print(json.dumps({s: r}), flush=True)
    hm.deallocate(); b.deallocate()
json.dump(res, open(a.out, "w"), indent=1)
