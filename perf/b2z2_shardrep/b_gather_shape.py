"""The `b` gather, priced at the two layouts it could be taken in, at identical bytes.

`b_shard` gathers `b` AFTER the channel move, so the collective is a concat on the LAST axis of
[1, c, S, S/N]. It could instead be taken before the move, on axis 1 (starting variant) or axis 2
(ending one) of [1, S/N, S, c] -- the same bytes, a different page shape, and the transform then
runs replicated. `b2z2-trunk-shard-scale-wh` measured 1.53x between two page shapes of the same
67.11 MB, so which of these is cheaper is a measurement and not a dataflow argument.

Every point is value-checked before it is believed: each device is handed a slab filled with its
own index and the gathered tensor must equal the full ramp on every device.

    MESH_N=4 TT_VISIBLE_DEVICES=28,29,30,31 BG_S=512 PYTHONPATH=$PWD python3 \
        perf/b2z2_shardrep/b_gather_shape.py
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
sys.path.insert(0, str(_HERE.parents[0] / "b2z2_shardscale"))

import torch  # noqa: E402
import meshdesc  # noqa: E402

N = int(os.environ.get("MESH_N", "4"))
S = int(os.environ.get("BG_S", "512"))
REPS = int(os.environ.get("BG_REPS", "9"))
OUT_PATH = os.environ.get("BG_OUT", f"/tmp/b2z2_bgather_{N}.json")
CHANNELS = [int(x) for x in os.environ.get("BG_C", "16,32,64,128").split(",")]

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
log(f"device={dev} arch={dev.arch()} N={N} S={S}")
COMP = ttnn.concat_mesh_to_tensor_composer(dev, 0)
RES = {"mesh_n": N, "S": S, "reps": REPS, "points": []}


def measure(label, full_shape, dim, c):
    """`full_shape` is what the gather must produce; it is split N ways on `dim`."""
    slab = list(full_shape)
    assert slab[dim] % N == 0
    slab[dim] //= N
    # Each device's slab carries its own index, so the gathered tensor is a ramp along `dim`
    # and a gather that returned the local slab N times cannot pass.
    ramp = torch.arange(full_shape[dim], dtype=torch.float32)
    view = [1] * len(full_shape)
    view[dim] = full_shape[dim]
    host = ramp.view(view).expand(full_shape).contiguous()
    t = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(dev, dim=dim))
    g = ttnn.all_gather(t, dim=dim)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(g, mesh_composer=COMP)
    per = got.shape[0] // N
    exact = all(torch.equal(got[i * per:(i + 1) * per], host.to(torch.bfloat16).float())
                for i in range(N))
    ttnn.deallocate(g)
    v = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        g = ttnn.all_gather(t, dim=dim)
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t0)
        ttnn.deallocate(g)
    v.sort()
    med = statistics.median(v)
    out_b = 1
    for d in full_shape:
        out_b *= d
    out_b *= 2
    moved = out_b * (N - 1) / N
    # What one trimul actually pays: this collective once per channel chunk of width c.
    per_trimul = med * (128 // c)
    RES["points"].append({"label": label, "shape": list(full_shape), "dim": dim, "c": c,
                          "exact": bool(exact), "out_bytes": out_b, "median_us": med * 1e6,
                          "gbps_per_dir": moved / med / 1e9,
                          "per_trimul_ms": per_trimul * 1e3})
    log(f"{label:28s} {str(list(full_shape)):24s} dim={dim} c={c:3d}  exact={exact}  "
        f"{med*1e6:9.1f} us  {moved/med/1e9:6.2f} GB/s/dir  "
        f"x{128//c} per trimul = {per_trimul*1e3:7.3f} ms")
    ttnn.deallocate(t)
    return exact


ok = True
for c in CHANNELS:
    # what b_shard does today: gather the MOVED chunk on its last axis, which is the output column
    ok &= measure("post-move, last axis", (1, c, S, S), 3, c)
    # the alternative: gather the gated projection on its row axis, transform replicated after
    ok &= measure("pre-move, row axis", (1, S, S, c), 1, c)
    # the ending variant's pre-move axis is the column one
    ok &= measure("pre-move, column axis", (1, S, S, c), 2, c)

tt.cleanup()
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
sys.exit(0 if ok else 1)
