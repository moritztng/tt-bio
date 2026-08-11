#!/usr/bin/env python3
"""Decompose the PRODUCTION TriangleMultiplication call at 512 aa, op by op.

The older `perf/trimul_kernel/opsplit298.py` replays a hand-copy of `__call__`; main has moved
(two-transpose channel move, `_transform_chunk`, one concat, row-blocked tail), and at N=512 that
replay is 50.81 ms against the module's own 28.48 ms, so its shares are not production's. This
instead wraps the ttnn entry points the module calls and runs the real module, so the tape is a
decomposition by construction. Every timed region has a device sync on both sides, which serialises
what the pipeline would overlap: the tape sum is therefore an upper bound and is compared against
the unsynced module wall in the output.
"""
import collections, json, statistics as st, time
from pathlib import Path
import sys

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
layer, c_z = build_layer(ckc)
tm = layer.triangle_multiplication_start
torch.manual_seed(0)
z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)

ROWS = []
ON = [False]


def shp(t):
    try:
        return "x".join(str(d) for d in t.shape)
    except Exception:
        return "?"


def buf(kw, t=None):
    mc = kw.get("memory_config")
    if mc is not None:
        return "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
    return "-"


def wrap(mod, name, tagger):
    orig = getattr(mod, name)

    def f(*a, **kw):
        if not ON[0]:
            return orig(*a, **kw)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        ttnn.synchronize_device(dev)
        ROWS.append((tagger(a, kw), (time.perf_counter() - t0) * 1e3, shp(a[0]) if a else "?"))
        return out
    setattr(mod, name, f)


wrap(ttnn, "matmul", lambda a, kw: f"matmul[{shp(a[0])}@{shp(a[1])}]->{buf(kw)}")
wrap(ttnn, "linear", lambda a, kw: f"linear[{shp(a[0])}@{shp(a[1])}]->{buf(kw)}")
wrap(ttnn.experimental, "minimal_matmul",
     lambda a, kw: f"minimal_matmul[{shp(a[0])}@{shp(a[1])}]->{buf(kw)}")
wrap(ttnn, "permute", lambda a, kw: f"permute{tuple(a[1])}[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "transpose", lambda a, kw: f"transpose({a[1]},{a[2]})[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "layer_norm", lambda a, kw: f"layer_norm[{shp(a[0])}]")
wrap(ttnn, "multiply_", lambda a, kw: f"multiply_[{shp(a[0])}]")
wrap(ttnn, "chunk", lambda a, kw: f"chunk{kw.get('chunks', '')}[{shp(a[0])}]")
wrap(ttnn, "concat", lambda a, kw: f"concat[{len(a[0])}x{shp(a[0][0])}]")
wrap(ttnn, "clone", lambda a, kw: f"clone[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "reallocate", lambda a, kw: f"reallocate[{shp(a[0])}]")
wrap(ttnn, "to_torch", lambda a, kw: f"to_torch[{shp(a[0])}]")
wrap(ttnn, "from_torch", lambda a, kw: "from_torch")
for op in ("reblock_permute", "reblock_permute_back"):
    if hasattr(ttnn.experimental, op):
        wrap(ttnn.experimental, op, lambda a, kw, op=op: f"{op}[{shp(a[0])}]->{buf(kw)}")


def once():
    return tm(z, None)


# untaped: the number every share is a share of
for _ in range(3):
    ttnn.deallocate(once())
ttnn.synchronize_device(dev)
ser = []
for _ in range(5):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    r = once()
    ttnn.synchronize_device(dev)
    ser.append((time.perf_counter() - t0) * 1e3)
    ttnn.deallocate(r)
ttnn.synchronize_device(dev)
t0 = time.perf_counter()
outs = [once() for _ in range(6)]
ttnn.synchronize_device(dev)
pipe = (time.perf_counter() - t0) * 1e3 / 6
for o in outs:
    ttnn.deallocate(o)
mod_ser, mod_pipe = st.median(ser), pipe
print(f"module: serial {mod_ser:.3f} ms  pipe {mod_pipe:.3f} ms", flush=True)

ON[0] = True
ttnn.deallocate(once())
ON[0] = False

agg = collections.OrderedDict()
for tag, ms, s0 in ROWS:
    a = agg.setdefault(tag, dict(tag=tag, n=0, ms=0.0))
    a["n"] += 1
    a["ms"] += ms
total = sum(a["ms"] for a in agg.values())
rows = sorted(agg.values(), key=lambda a: -a["ms"])
print(f"taped sum {total:.3f} ms against module pipe {mod_pipe:.3f} ms "
      f"(sync inflation {total / mod_pipe:.3f}x)")
print(f"{'op':64s} {'n':>3s} {'ms':>8s} {'%':>6s}")
for a in rows:
    a["ms"] = round(a["ms"], 4)
    a["share"] = round(a["ms"] / total, 4)
    print(f"{a['tag']:64s} {a['n']:3d} {a['ms']:8.3f} {100 * a['share']:5.1f}%")

res = dict(n=N, c_z=c_z, hidden=tm._hidden, grid=list(COMPUTE_GRID_MAIN),
           module_serial_ms=round(mod_ser, 4), module_pipe_ms=round(mod_pipe, 4),
           taped_sum_ms=round(total, 4), sync_inflation=round(total / mod_pipe, 4), ops=rows)
Path(f"perf/trimul_root/tape_{N}_qb2c0.json").write_text(json.dumps(res, indent=2))
print("RESULT_JSON " + json.dumps(res), flush=True)
