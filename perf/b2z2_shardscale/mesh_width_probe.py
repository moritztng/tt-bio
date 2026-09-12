"""Does a 1xN line of galaxy chips come up as a mesh, and does its all_gather return the right bytes?

`b2z2-pairtrack-chain-wh` brought up a 1x2 mesh on whglx with the stock n300 descriptor. Nothing on
this campaign has ever asked for four or eight. This is the cheapest possible question -- open the
mesh, gather one small tensor, check every device got every other device's rows and not its own
twice, close -- and it is asked first because a width that cannot come up must not be discovered
inside a forty-minute block sweep.

The check reads VALUES, not shapes: each device is handed a slab filled with its own index, so a
gather that returned the local slab N times, or the slabs in the wrong order, is rejected. A shape
check would pass on both.

    MESH_N=4 TT_VISIBLE_DEVICES=16,17,18,19 TT_BIO_LEASE_CARDS=16,17,18,19 \
    TT_BIO_LEASE_HOLDER=worker:b2z2-trunk-shard-scale-wh \
    PYTHONPATH=$PWD python3 perf/b2z2_shardscale/mesh_width_probe.py

Exit status 0 only if the mesh opened at the requested width and the gather was exact.
"""

import json
import os
import pathlib
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
ROWS = int(os.environ.get("MESH_ROWS", "1"))
OUT_PATH = os.environ.get("PROBE_OUT", f"/tmp/b2z2_meshprobe_{N}.json")

DESC = meshdesc.install(N, rows=ROWS) if N > 1 else None

import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log(f"MESH_N={N} rows={ROWS} visible={os.environ.get('TT_VISIBLE_DEVICES')} desc={DESC}")
OPENS = [0]


def _mesh_open(device_id, kwargs):
    OPENS[0] += 1
    with tt._device_init_lock():
        if N > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        log(f"opening {ROWS}x{N // ROWS} mesh, kwargs={kwargs}")
        dev = (ttnn.open_mesh_device(ttnn.MeshShape(ROWS, N // ROWS), **kwargs) if N > 1
               else ttnn.open_device(device_id=device_id, **kwargs))
        tt._configure_active_compute_grid(dev)
        dev.enable_program_cache()
        return dev


tt._open_device_locked = _mesh_open
t_open = time.perf_counter()
dev = tt.get_device()
t_open = time.perf_counter() - t_open
log(f"device={dev} arch={dev.arch()} grid={tt.CORE_GRID_MAIN} open took {t_open:.1f} s")
assert OPENS[0] == 1, "the mesh was opened more than once"

RES = {"mesh_n": N, "rows": ROWS, "visible": os.environ.get("TT_VISIBLE_DEVICES"),
       "arch": str(dev.arch()), "open_s": round(t_open, 2), "desc": DESC}
fail = []

ROWS_PER, C = 64, 128
if N > 1:
    # Each device's slab is filled with its own index, so the gathered tensor must be
    # [0]*64 ++ [1]*64 ++ ... -- a local-slab-repeated gather gives [k]*64 N times and is caught.
    host = torch.cat([torch.full((1, ROWS_PER, C), float(i)) for i in range(N)], dim=1)
    shard = ttnn.shard_tensor_to_mesh_mapper(dev, dim=1)
    comp = ttnn.concat_mesh_to_tensor_composer(dev, 0)
    t = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        mesh_mapper=shard)
    log(f"sharded {tuple(host.shape)} -> per device {tuple(t.shape)}")
    t0 = time.perf_counter()
    g = ttnn.all_gather(t, dim=1)
    got = ttnn.to_torch(g, mesh_composer=comp)
    dt = time.perf_counter() - t0
    per_dev = [torch.equal(got[i:i + 1], host) for i in range(N)]
    RES["gather"] = {"per_device_exact": per_dev, "shape": list(got.shape),
                     "max_abs_diff": float((got - host.repeat(N, 1, 1)).abs().max()),
                     "cold_call_ms": round(dt * 1e3, 2)}
    log(f"all_gather: per-device exact {per_dev}  cold {dt * 1e3:.1f} ms")
    if not all(per_dev):
        fail.append(f"the 1x{N} all_gather did not return every device's rows to every device")
    ttnn.deallocate(g)
    ttnn.deallocate(t)
else:
    x = ttnn.from_torch(torch.randn(1, ROWS_PER, C), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    RES["gather"] = {"single_chip": True, "shape": list(x.shape)}
    ttnn.deallocate(x)

tt.cleanup()
if N > 1:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")

if fail:
    for f in fail:
        print(f"  - {f}")
    log(f"FAIL at width {N}")
    sys.exit(1)
log(f"PASS: a 1x{N} mesh came up on {os.environ.get('TT_VISIBLE_DEVICES')} and its gather is exact")
sys.exit(0)
