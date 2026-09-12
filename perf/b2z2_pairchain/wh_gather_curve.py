"""What an `all_gather` costs on the Wormhole pair, measured at this box's own small end.

CONTEXT §4-D: the p300c fit `t = 10.7 us + bytes/20.1 GB/s` was fitted over 4.2-134 MB and
under-prices anything below ~2 MB by up to 10x, because the collective is latency-bound there and
the real fixed term is ~110 us. That is a Blackhole number on a Blackhole card. This measures the
Wormhole galaxy pair from 0.26 MB to 67.1 MB so the two regimes separate here rather than being
inherited, and every size is verified bit-exact on both devices so a gather that returned its own
shard cannot pass as a transfer.

    MESH_N=2 TT_VISIBLE_DEVICES=12,13 python3 perf/b2z2_pairchain/wh_gather_curve.py
"""

import json
import os
import pathlib
import statistics as st
import sys
import time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

OUT_PATH = os.environ.get("CURVE_OUT", "/tmp/b2z2_wh_gather_curve.json")
BURST = int(os.environ.get("CURVE_BURST", "20"))

import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _mesh_open(device_id, kwargs):
    with tt._device_init_lock():
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), **kwargs)
        tt._configure_active_compute_grid(dev)
        dev.enable_program_cache()
        return dev


tt._open_device_locked = _mesh_open
mesh = tt.get_device()
log(f"mesh={mesh} arch={mesh.arch()}")

RES = {"arch": str(mesh.arch()), "host": os.uname().nodename,
       "visible": os.environ.get("TT_VISIBLE_DEVICES"), "burst": BURST, "sizes": []}
COMP = ttnn.concat_mesh_to_tensor_composer(mesh, 0)

for I in (2, 4, 8, 16, 32, 64, 128, 256, 512):
    host = torch.arange(I * 512 * 128, dtype=torch.int32).remainder(1021).to(torch.bfloat16)
    host = host.reshape(1, I, 512, 128)
    nbytes = host.numel() * 2
    t = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, 1))
    g = ttnn.all_gather(t, dim=1)
    ttnn.synchronize_device(mesh)
    # Is it a transfer? Both devices must hold the WHOLE tensor, including the half they never had.
    back = ttnn.to_torch(g, mesh_composer=COMP)
    exact = [bool(torch.equal(back[i:i + 1].float(), host.float())) for i in range(2)]
    ttnn.deallocate(g)

    samples = []
    for _ in range(BURST):
        t0 = time.perf_counter()
        g = ttnn.all_gather(t, dim=1)
        ttnn.synchronize_device(mesh)
        samples.append(time.perf_counter() - t0)
        ttnn.deallocate(g)
    med = st.median(samples)
    row = {"i_axis": I, "bytes": nbytes, "bytes_per_direction": nbytes / 2,
           "median_s": med, "min_s": min(samples), "max_s": max(samples),
           "us": med * 1e6, "gbps_per_direction": nbytes / 2 / med / 1e9,
           "bit_exact_both_devices": exact}
    RES["sizes"].append(row)
    log(f"  I={I:4d}  {nbytes / 1e6:7.2f} MB  {med * 1e6:9.1f} us  "
        f"{row['gbps_per_direction']:6.2f} GB/s/dir  bit-exact {exact}")
    ttnn.deallocate(t)

# Two-regime fit, with the crossover read off the data rather than assumed: the small end is flat
# in size (latency-bound) and the large end linear in size (bandwidth-bound).
big = [r for r in RES["sizes"] if r["bytes"] >= 4 * 2 ** 20]
small = [r for r in RES["sizes"] if r["bytes"] < 4 * 2 ** 20]
if len(big) >= 2:
    x = [r["bytes"] for r in big]
    y = [r["median_s"] for r in big]
    n = len(x)
    sx, sy = sum(x), sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(a * b for a, b in zip(x, y))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    inter = (sy - slope * sx) / n
    RES["fit_large"] = {"over_bytes": [min(x), max(x)], "intercept_us": inter * 1e6,
                        "gbps": 1 / slope / 1e9}
    log(f"large-size fit ({min(x) / 1e6:.1f}-{max(x) / 1e6:.1f} MB): "
        f"t = {inter * 1e6:.1f} us + bytes / {1 / slope / 1e9:.1f} GB/s")
if small:
    RES["fit_small"] = {"median_us": st.median([r["us"] for r in small]),
                        "range_us": [min(r["us"] for r in small), max(r["us"] for r in small)],
                        "over_bytes": [min(r["bytes"] for r in small),
                                       max(r["bytes"] for r in small)]}
    log(f"small end ({min(r['bytes'] for r in small) / 1e6:.2f}-"
        f"{max(r['bytes'] for r in small) / 1e6:.2f} MB): "
        f"{RES['fit_small']['range_us'][0]:.1f}-{RES['fit_small']['range_us'][1]:.1f} us, "
        f"flat in size = latency-bound")

tt.cleanup()
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
