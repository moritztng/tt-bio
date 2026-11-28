"""Does a per-pose batched scatter beat the dense one, and is it right? Card probe, no model code.

The static graph tiles one pose's edges over P poses, so the dense [P*n, P*e] 0/1 scatter is P
copies of one [n, e] block on the diagonal. Per edge block of the gated ligand (5SB2_1K2, P=10):
time ``S_dense @ msg`` against ``S_block[P, n, e] @ msg[P, e, d]`` (and the same with one shared
[1, n, e] block, if matmul broadcasts it), and grade both against a float64 torch scatter.
"""
import argparse, json, time
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()
import torch
import ttnn
from tt_bio.diffdock import device as ddev, network
from tt_bio.diffdock.job import DockOptions
model = network.load(DockOptions(device="tt", edge_tp="tile"))
layer = next(iter(model.convs.values())).dev
dev, ckc = layer.device, layer.ckc
P, D = 10, 160
BLOCKS = {"lig<-lig": (32, 960), "lig<-cross": (32, 9920), "rec<-rec": (352, 7936),
          "rec<-cross": (352, 9920)}
torch.manual_seed(0)
up = lambda t, dt=ttnn.float32: ttnn.from_torch(t.float(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
s_dt = ddev.exact01_dtype(dev, ttnn.float32)


def timed(fn):
    fn(); ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(a.reps):
        r = fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / a.reps * 1e3, r


def rel(x, ref):
    return float((x.double() - ref).norm() / ref.norm())


res = {"P": P, "D": D, "reps": a.reps, "blocks": {}}
for name, (n, e) in BLOCKS.items():
    src = torch.randint(0, n, (e,))
    blk = torch.zeros(n, e); blk[src, torch.arange(e)] = 1.0
    msg = torch.randn(P * e, D)
    ref = torch.einsum("ne,ped->pnd", blk.double(), msg.double().reshape(P, e, D))
    row = {"n": n, "e": e}
    m2 = up(msg)
    try:
        dense = up(torch.block_diag(*[blk] * P), s_dt)
        ms, r = timed(lambda: ttnn.matmul(dense, m2, compute_kernel_config=ckc, dtype=ttnn.float32))
        row["dense_ms"] = round(ms, 3)
        row["dense_rel_err"] = rel(ttnn.to_torch(r).reshape(P, n, D), ref)
        ttnn.deallocate(dense)
    except Exception as ex:
        row["dense_error"] = repr(ex)[:300]
    m3 = ttnn.reshape(m2, (P, e, D))
    for label, s in (("batched", blk.expand(P, n, e)), ("shared", blk.reshape(1, n, e))):
        try:
            st = up(s, s_dt)
            ms, r = timed(lambda: ttnn.matmul(st, m3, compute_kernel_config=ckc, dtype=ttnn.float32))
            row[label + "_ms"] = round(ms, 3)
            row[label + "_rel_err"] = rel(ttnn.to_torch(r).reshape(P, n, D), ref)
            ttnn.deallocate(st)
        except Exception as ex:
            row[label + "_error"] = repr(ex)[:300]
    try:
        ms, _ = timed(lambda: ttnn.reshape(m2, (P, e, D)))
        row["reshape_ms"] = round(ms, 3)
    except Exception as ex:
        row["reshape_error"] = repr(ex)[:300]
    res["blocks"][name] = row
    print(name, row, flush=True)
    open(a.out, "w").write(json.dumps(res, indent=1))
