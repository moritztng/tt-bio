"""ESMFold2 confidence expectations on the device (esmfold2_runtime._expectations_dev) vs float64.

Random [1,L,L,64] logits with a pae-like spread, rounded to bf16 as the head produces them. Reports max and mean
abs error of PAE (A), PDE (A) and the TM term against float64 on the same bf16 logits, for the device reduction and
for today's path (fp32 download + host _pair_expectations), and the wall time of each.

    python perf/spd_gap_esmf/conf_proj.py --ns 384,512 --out r.jsonl
"""
import argparse, json, time

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="512")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio.af2 import compute_kernel_config
from tt_bio.esmfold2_runtime import _expectations_dev
from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import _pair_expectations, _tm_per_bin

dev = T.get_device()
ck = T.trunk_compute_kernel_config(compute_kernel_config())


def timed(fn):
    best = None
    for _ in range(A.reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter(); r = fn(); ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return r, best * 1e3


for n in (int(x) for x in A.ns.split(",")):
    g = torch.Generator().manual_seed(n)
    lg = (torch.randn(1, n, n, 64, generator=g) * 3 + torch.linspace(4, -4, 64)).bfloat16().float()
    tm = _tm_per_bin(torch.ones(1, n), 64)
    c = torch.linspace(0.0, 32.0, 65); c = (c[:-1] + c[1:]) / 2
    p = torch.softmax(lg.double(), -1)
    ref = torch.stack([p @ c.double(), p @ tm[0].double()], -1)
    lg_d = ttnn.from_torch(lg, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    up = lambda *cols: ttnn.from_torch(torch.stack(cols, -1).float().reshape(1, 1, 64, len(cols)),
                                       layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    v = up(c, tm[0], torch.ones(64))
    dv, dev_ms = timed(lambda: _expectations_dev(lg_d, v, ck))

    def host():
        x = torch.Tensor(ttnn.to_torch(lg_d)).float()
        pae, pde, tme = _pair_expectations(x, x, tm)
        return torch.stack([pae, tme], -1)
    hv, host_ms = timed(host)
    row = {"n": n, "dev_ms": round(dev_ms, 2), "host_ms": round(host_ms, 2)}
    for tag, got in (("dev", dv), ("host", hv)):
        e = (got.double() - ref).abs()
        row[f"{tag}_pae_max_A"] = round(float(e[..., 0].max()), 5)
        row[f"{tag}_pae_mean_A"] = round(float(e[..., 0].mean()), 6)
        row[f"{tag}_tm_max"] = round(float(e[..., 1].max()), 6)
    print(json.dumps(row), flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    ttnn.deallocate(lg_d); ttnn.deallocate(v)
T.cleanup()
