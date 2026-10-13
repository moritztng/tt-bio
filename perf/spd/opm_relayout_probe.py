"""OPM relayout: today's chain against one tile reshape, at OpenFold3's 512-token shape.

The contraction z = a_flat @ b^T with a_flat = [(i c), S] gives [(i c), (d j)] today (b's rows are
(d j)), and four ops turn it into the projection's [i, j, (c d)] (cen68: 11.5 ms/call, 16 calls a fold).
With b's rows ordered (j d) instead, z's tile (i, j) holds exactly the [c, d] block of pair (i, j), so
z's bytes are those of a tiled [I, J, C, D] tensor and the projection input is that tensor reshaped to
[I*J, C*D]. This times each candidate and checks it against torch, so the model change can be chosen
on a measurement:
  chain    - the shipped TILE->RM, reshape, RM->TILE, permute from [(i c), (d j)]
  tresh    - ttnn.reshape of a tiled [I, J, C, D] to [I*J, C*D] (input built directly in that shape)
  view     - whether ttnn can re-label z's [(i c), (j d)] as [I, J, C, D] without a move

Run: TT_VISIBLE_DEVICES=<card> timeout 600 python perf/spd/opm_relayout_probe.py --out r.json
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument("--I", type=int, default=512)
ap.add_argument("--C", type=int, default=32)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
I = J = a.I
C = D = a.C
dev = T.get_device()
res = {"I": I, "C": C, "aiclk": []}


def clk():
    try:
        for n in Path("/sys/class/tenstorrent").iterdir():
            res["aiclk"].append(int((n / "tt_aiclk").read_text()))
    except Exception:
        pass


def put(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def timed(name, make, fn, want):
    ms, out = [], None
    for r in range(a.reps):
        x = make()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        y = fn(x)
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
        clk()
        if r == a.reps - 1:
            out = ttnn.to_torch(y).float()
        ttnn.deallocate(y)
    ok = bool(torch.isfinite(out).all()) and torch.equal(out.reshape(want.shape), want)
    res[name] = {"ms": round(statistics.median(ms[1:]), 3), "exact": ok}
    print(name, res[name], flush=True)


torch.manual_seed(0)
blk = torch.randn(I, J, C, D).bfloat16()               # blk[i, j, c, d] = outer-product block
want = blk.reshape(I, J, C * D).float()

# chain: z [(i c), (d j)] as today
z_old = blk.permute(0, 2, 3, 1).reshape(I * C, D * J)


def chain(z):
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    z = ttnn.reshape(z, (I, C * D, J))
    z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
    return ttnn.permute(z, (0, 2, 1))


try:
    timed("chain", lambda: put(z_old), chain, want)
except Exception as e:
    res["chain"] = {"error": repr(e)[:300]}

try:
    timed("tresh", lambda: put(blk), lambda z: ttnn.reshape(z, (I, J, C * D)), want)
except Exception as e:
    res["tresh"] = {"error": repr(e)[:300]}

# view: the (j d) contraction's z, [(i c), (j d)]; are its bytes a tiled [I, J, C, D]?
z_new = blk.permute(0, 2, 1, 3).reshape(I * C, J * D)
try:
    zt = put(z_new)
    v = None
    for f in ("view", "reshape"):
        g = getattr(getattr(ttnn, "experimental", None), f, None) if f == "view" else None
        if g is not None:
            try:
                v = g(zt, ttnn.Shape([I, J, C, D]))
                res["view_api"] = "ttnn.experimental.view"
                break
            except Exception as e:
                res["view_err"] = repr(e)[:300]
    res["view_exact"] = None if v is None else torch.equal(ttnn.to_torch(v).float(), blk.float())
except Exception as e:
    res["view_err2"] = repr(e)[:300]

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res))
print(json.dumps({k: v for k, v in res.items() if k != "aiclk"}))
