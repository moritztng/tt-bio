"""The ending triangle attention's two pair transposes at OpenDDE's c_z 384, against their reader-shuffle split.

cs5 (9LLG, 640 tokens, WH 1000 MHz) prices `pair_transpose` [640,640,384] at 5.2 ms (121 GB/s) and
`transpose_add` at 6.9 ms (137 GB/s), 480 calls each a fold. Both split the 32-tile row gather between
reader and writer at QR 8, tuned on Protenix-v2's c_z 256 (175 GB/s at 1184). This sweeps QR per op and
shape. Every arm is checked torch.equal against the torch transpose (pair_transpose) or the QR 8 arm
(transpose_add).

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/ptr_sweep.py OUT.json [--sizes 512 640] [--qr 0 2 4 ...]
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--sizes", nargs="+", type=int, default=[512, 640])
ap.add_argument("--c", type=int, default=384)
ap.add_argument("--qr", nargs="+", type=int, default=[0, 2, 4, 6, 8, 10, 12, 16])
ap.add_argument("--ncall", type=int, default=5); ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import pair_transpose as PT, pair_transpose_add as PTA

dev = T.get_device()
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)


def timed(fn, after=None):
    ts = []
    for _ in range(a.reps):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(a.ncall):
            o = fn()
            if after:
                after(o)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / a.ncall * 1e3)
    return statistics.median(ts), ts


res = {"arch": str(dev.arch()), "rows": []}
for S in a.sizes:
    torch.manual_seed(S)
    ut, zt = torch.randn(S, S, a.c).bfloat16(), torch.randn(S, S, a.c).bfloat16()
    u = up(ut)
    pbytes = S * S * a.c * 2
    ref_t = ut.transpose(0, 1).contiguous()
    base_add = None
    for qr in a.qr:
        PT.PAIR_TRANSPOSE_SPLIT = qr
        PT._CACHE.clear()
        row = {"S": S, "op": "pair_transpose", "qr": qr}
        try:
            o = PT.pair_transpose(u, ttnn.DRAM_MEMORY_CONFIG)
            row["equal"] = bool(torch.equal(ttnn.to_torch(o).reshape(S, S, a.c), ref_t)); o.deallocate()
            row["ms"], row["ms_all"] = timed(lambda: PT.pair_transpose(u, ttnn.DRAM_MEMORY_CONFIG), lambda o: o.deallocate())
            row["gbs"] = 2 * pbytes / row["ms"] / 1e6
        except Exception as e:
            row["error"] = str(e).splitlines()[0][:200]
        print(row, flush=True); res["rows"].append(row)

        PTA.QR = qr
        PTA._CACHE.clear()
        row = {"S": S, "op": "transpose_add", "qr": qr}
        try:
            z = up(zt)
            PTA.transpose_add(u, z)
            got = ttnn.to_torch(z).reshape(S, S, a.c); z.deallocate()
            if base_add is None:
                base_add = got
            row["equal"] = bool(torch.equal(got, base_add))
            z = up(zt)
            row["ms"], row["ms_all"] = timed(lambda: PTA.transpose_add(u, z))
            z.deallocate()
            row["gbs"] = 3 * pbytes / row["ms"] / 1e6
        except Exception as e:
            row["error"] = str(e).splitlines()[0][:200]
        print(row, flush=True); res["rows"].append(row)
    u.deallocate()
json.dump(res, open(a.out, "w"), indent=1)
