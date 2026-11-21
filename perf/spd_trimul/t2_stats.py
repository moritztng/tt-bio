#!/usr/bin/env python3
"""Producer 2 of T1's forwarded LayerNorm statistics: RFD4's trimul tail T2 on the bf16 pair stream
(`fused_tail_ln(stream16=True)`, hidden K -> c_z C) timed without and with `stats`, AICLK sampled during the calls.
stats_ms - base_ms is the producer's cost. The stats are then checked against float64 statistics of the z the
program wrote (dev_*) and beside them T1's own computation, the bf16-truncated 1/C scaler (t1emu_*), as or_head.py.
The written z must be bit-identical with and without stats (z_identical).

    python perf/spd_trimul/t2_stats.py --ns 352,256 --out t2st.jsonl
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # perf/ for clocksample
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="352,256")
ap.add_argument("--k", type=int, default=256, help="trimul hidden (x channels)")
ap.add_argument("--c", type=int, default=384, help="c_z (z channels)")
ap.add_argument("--mu", type=float, default=3.0, help="per-row offset of z, so the mean matters")
ap.add_argument("--eps", type=float, default=1e-5)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config
import clocksample

dev = T.get_device()
ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
grid = tuple(T.COMPUTE_GRID_MAIN)
K, C = A.k, A.c
gen = torch.Generator().manual_seed(0)
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


for n in (int(v) for v in A.ns.split(",")):
    z0 = torch.randn(1, n, n, C, generator=gen) + A.mu * torch.randn(1, n, n, 1, generator=gen)
    d = {"x": up(torch.randn(1, n, n, K, generator=gen)), "g": up(torch.randn(1, n, n, C, generator=gen)),
         "z": up(z0), "w": up(torch.randn(K, C, generator=gen) / K ** 0.5),
         "gamma": up(1 + 0.1 * torch.randn(K, generator=gen)), "beta": up(0.1 * torch.randn(K, generator=gen))}
    st = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n // 32, 64]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, dev,
                                        ttnn.DRAM_MEMORY_CONFIG)
    rec = {"n": n, "k": K, "c": C}
    call = lambda s: TT.fused_tail_ln(d["x"], d["gamma"], d["beta"], d["w"], d["g"], d["z"], ckc, grid,  # noqa: E731
                                      eps=A.eps, stream16=True, stats=s)
    if call(None) is None or call(st) is None:
        rec["err"] = "declined"
    else:
        clk = clocksample.during(period=0.5).__enter__()
        rec["base_ms"] = timed(lambda: call(None), A.reps)
        rec["stats_ms"] = timed(lambda: call(st), A.reps)
        clk.__exit__(None, None, None)
        rec["aiclk"] = clk.summary()
        # one more call each from the same z: the written z must not depend on whether stats are written
        zin = ttnn.to_torch(d["z"])
        call(None); za = ttnn.to_torch(d["z"])
        ttnn.deallocate(d["z"]); d["z"] = up(zin)
        call(st); zb = ttnn.to_torch(d["z"])
        rec["z_identical"] = bool(torch.equal(za, zb))
        x = zb.double().reshape(n, n, C)
        s = ttnn.to_torch(st).double().reshape(n, n // 32, 64)
        dm, dr = s[..., :32].reshape(n, n), s[..., 32:].reshape(n, n)
        mu, var = x.mean(-1), x.var(-1, unbiased=False)
        sd, rs = var.sqrt(), (var + A.eps).rsqrt()
        inv = torch.tensor([1.0 / C], dtype=torch.float32).view(torch.int32) & -65536
        k = float(inv.view(torch.float32))                                   # T1's bf16-truncated 1/C
        em = x.sum(-1) * k
        er = ((x - em[..., None]).pow(2).sum(-1) * k + A.eps).rsqrt()
        for tag, m_, r_ in (("dev", dm, dr), ("t1emu", em, er)):
            me, re = (m_ - mu) / sd, r_ / rs - 1
            rec[f"{tag}_mean_err_rms"] = float(me.pow(2).mean().sqrt())
            rec[f"{tag}_mean_err_max"] = float(me.abs().max())
            rec[f"{tag}_rstd_rel_rms"] = float(re.pow(2).mean().sqrt())
            rec[f"{tag}_rstd_rel_max"] = float(re.abs().max())
            rec[f"{tag}_rstd_rel_mean"] = float(re.mean())
    ttnn.deallocate(st)
    for t in d.values():
        ttnn.deallocate(t)
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as fh:
            fh.write(line + "\n")
