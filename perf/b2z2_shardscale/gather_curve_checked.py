"""The all_gather curve at width N, with every point's VALUES checked before its time is believed.

This row measures 34.5 GB/s per direction on chips 22+23 where `b2z2-pairtrack-chain-wh` measured
21.7 on chips 12+13 of the same box. A link that came out 1.6x faster than the row next door is
exactly the shape of a measurement that is timing nothing, so every size here is verified first:
each device is handed a slab filled with its own index, and the gathered tensor must equal the
full ramp on EVERY device. A gather that returned the local slab N times, or the slabs in the
wrong order, is rejected -- and a gather that moved no bytes cannot produce the ramp at all.

    MESH_N=8 TT_VISIBLE_DEVICES=16,...,23 PYTHONPATH=$PWD \
    python3 perf/b2z2_shardscale/gather_curve_checked.py

Exit status 0 only if every size was exact on every device.
"""

import json
import os
import pathlib
import statistics
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = str(_HERE.parents[1])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, str(_HERE))

import torch  # noqa: E402
import meshdesc  # noqa: E402

N = int(os.environ.get("MESH_N", "2"))
REPS = int(os.environ.get("GC_REPS", "20"))
OUT_PATH = os.environ.get("GC_OUT", f"/tmp/b2z2_gathercurve_{N}.json")
C = 128

meshdesc.install(N)

import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _mesh_open(device_id, kwargs):
    with tt._device_init_lock():
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, N), **kwargs)
        tt._configure_active_compute_grid(dev)
        dev.enable_program_cache()
        return dev


tt._open_device_locked = _mesh_open
dev = tt.get_device()
log(f"device={dev} arch={dev.arch()} N={N} visible={os.environ.get('TT_VISIBLE_DEVICES')}")

RES = {"mesh_n": N, "arch": str(dev.arch()), "reps": REPS,
       "visible": os.environ.get("TT_VISIBLE_DEVICES"), "points": []}
fail = []

shard_map = ttnn.shard_tensor_to_mesh_mapper(dev, dim=1)
comp = ttnn.concat_mesh_to_tensor_composer(dev, 0)

# Row counts, not byte targets: a slab has to be a whole number of tiles on every device, so every
# size is a multiple of 32*N rows. The block's own pair-track gather at 512 aa is 512*512 rows of
# 128 channels = 67.11 MB, which is the 262144-row point.
ROWS = [r for r in (2048, 8192, 16384, 65536, 262144, 524288, 1048576) if r % (32 * N) == 0]

for rows in ROWS:
    host = torch.cat([torch.full((1, rows // N, C), float(i)) for i in range(N)], dim=1)
    t = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        mesh_mapper=shard_map)
    g = ttnn.all_gather(t, dim=1)
    got = ttnn.to_torch(g, mesh_composer=comp)
    exact = [torch.equal(got[i:i + 1], host) for i in range(N)]
    ttnn.deallocate(g)
    ttnn.synchronize_device(dev)
    v = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        g = ttnn.all_gather(t, dim=1)
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t0)
        ttnn.deallocate(g)
    v.sort()
    med = statistics.median(v)
    out_b = rows * C * 2
    moved = out_b * (N - 1) / N          # a ring gather lands (N-1)/N of the output on each device
    RES["points"].append({"rows": rows, "out_bytes": out_b, "per_device_in_bytes": out_b // N,
                          "moved_bytes": moved, "median_us": med * 1e6,
                          "min_us": v[0] * 1e6, "max_us": v[-1] * 1e6,
                          "gbps_per_dir": moved / med / 1e9, "exact_per_device": exact})
    log(f"out {out_b/1e6:8.2f} MB  in {out_b/N/1e6:7.3f} MB/dev  {med*1e6:9.1f} us  "
        f"{moved/med/1e9:6.2f} GB/s/dir  exact={all(exact)}")
    if not all(exact):
        fail.append(f"{out_b/1e6:.2f} MB: the gather did not return every device's rows")
    ttnn.deallocate(t)

# --- the shape the block actually gathers -------------------------------------------------------
# The curve above is rank 3, [1, rows, 128]. `_pair_track_row_sharded` gathers the PAIR TENSOR,
# [1, S/N, S, 128] on dim 1 -- same bytes, different pages. `b2z2-pairtrack-chain-wh` measured
# 1546.5 us for the 67.11 MB pair-track gather on chips 12+13 where the rank-3 curve here says
# 979.5 us on 22+23, so the page shape is the first thing that could explain a 1.6x gap between
# two rows on one box. This prices the real thing and settles it.
RES["pair_shape"] = []
for S in (256, 512, 768):
    if S % N:
        continue
    host = torch.cat([torch.full((1, S // N, S, C), float(i)) for i in range(N)], dim=1)
    t4 = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(dev, dim=1))
    g = ttnn.all_gather(t4, dim=1)
    got = ttnn.to_torch(g, mesh_composer=comp)
    exact = [torch.equal(got[i:i + 1], host) for i in range(N)]
    ttnn.deallocate(g)
    ttnn.synchronize_device(dev)
    v = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        g = ttnn.all_gather(t4, dim=1)
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t0)
        ttnn.deallocate(g)
    v.sort()
    med = statistics.median(v)
    out_b = S * S * C * 2
    moved = out_b * (N - 1) / N
    RES["pair_shape"].append({"S": S, "shape": [1, S, S, C], "out_bytes": out_b,
                              "median_us": med * 1e6, "gbps_per_dir": moved / med / 1e9,
                              "exact_per_device": exact})
    log(f"pair tensor S={S:4d} [1,{S},{S},{C}]  {out_b/1e6:7.2f} MB  {med*1e6:9.1f} us  "
        f"{moved/med/1e9:6.2f} GB/s/dir  exact={all(exact)}")
    if not all(exact):
        fail.append(f"S={S} pair-shape gather did not return every device's rows")
    ttnn.deallocate(t4)

# A fit over the points that are bandwidth-bound, and the fixed term read off the small end rather
# than extrapolated down to it -- CONTEXT 4-D: the p300c's published intercept is 10x too small
# because it was fitted at the large end only.
big = [p for p in RES["points"] if p["out_bytes"] >= 8e6]
if len(big) >= 2:
    xs = [p["moved_bytes"] for p in big]
    ys = [p["median_us"] * 1e-6 for p in big]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    RES["fit_large"] = {"intercept_us": a * 1e6, "gbps_per_dir": 1 / b / 1e9,
                        "over_mb": [min(xs) / 1e6, max(xs) / 1e6]}
    log(f"fit over {min(xs)/1e6:.1f}-{max(xs)/1e6:.1f} MB moved: "
        f"t = {a*1e6:.1f} us + moved / {1/b/1e9:.1f} GB/s")
small = [p for p in RES["points"] if p["out_bytes"] < 8e6]
if small:
    RES["latency_floor_us"] = min(p["median_us"] for p in small)
    log(f"latency floor at the small end: {RES['latency_floor_us']:.1f} us")

tt.cleanup()
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
if fail:
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
log(f"PASS: every point on the 1x{N} gather curve was exact on every device")
