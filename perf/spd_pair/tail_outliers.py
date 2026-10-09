"""Per-element wrong-value count for triangle attention's tail on the device, vs float64.

rel_rms and a structure grade cannot see a handful of elements off by exactly 1, 2 or 4 (the Wormhole
fp32-DST K-block erratum, BOARD 2026-10-09 12:08Z). This counts them. Inputs are random at the fold's
scales; the reference is float64 on the device's own bf16 inputs. Arms: the three-op path the fold
runs without `triatt_tail` (at the normal and the fast config), and the fused tail with and without
the residual, K in one DST block (KB0) or one tile per pass summed by the packer (KB1). The fused tail
runs HiFi3 with fp32 DST whatever the mode's config, so one fused count covers both modes.

usage: tail_outliers.py OUT --chip C [--sizes 384,512,736] [--seeds 0,1,2,3] [--cz 256] [--thr 0.1]
"""
import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--chip", type=int)
ap.add_argument("--sizes", default="384,512,736")
ap.add_argument("--seeds", default="0,1,2,3")
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--thr", type=float, default=0.1)
A = ap.parse_args()
OUT = Path(A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "outliers.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time()
    s = json.dumps(kw); LOG.write(s + "\n"); LOG.flush(); print(s, flush=True)


if A.chip is not None:
    from tt_bio import runtime, worker as W
    from tt_bio.host_controller import worker_payload
    slot = runtime.build_local_workers("tenstorrent", [object()], [A.chip])[0]
    W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.triatt_qkv as TQ

torch.set_num_threads(16)
dev = T.get_device()
fds = [os.readlink(f"/proc/self/fd/{f}") for f in os.listdir("/proc/self/fd") if os.path.exists(f"/proc/self/fd/{f}")]
nodes = sorted(int(p.rsplit("/", 1)[1]) for p in fds if p.startswith("/dev/tenstorrent/") and p.rsplit("/", 1)[1].isdigit())
NODE = nodes[0] if nodes else -1
log(ev="open", arch=str(dev.arch()), nodes=nodes, args=vars(A))
TQ.set_tail("1")


def ckc(fid, f32):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                            fp32_dest_acc_en=f32, packer_l1_acc=False)


CFG = {"normal": ckc("HiFi3", True), "fast": ckc("HiFi4", False)}


def count(got, ref, name, **kw):
    err = (got.double().reshape(ref.shape) - ref).abs()
    bad = err > A.thr
    rec = dict(arm=name, n=int(bad.sum()), n_gt05=int((err > 0.5).sum()), max_abs=float(err.max()),
               nonfinite=int((~torch.isfinite(got)).sum()), elements=ref.numel(), **kw)
    if rec["n"]:
        idx = bad.flatten().nonzero()[:8].flatten()
        rec["worst"] = [[int(i), float(ref.flatten()[i]), float(got.double().flatten()[i])] for i in idx]
    log(ev="count", **rec)
    return rec


C = A.cz
H = C // 32
for S in [int(v) for v in A.sizes.split(",")]:
    for seed in [int(v) for v in A.seeds.split(",")]:
        torch.manual_seed(seed)
        o_h = (torch.randn(S, H, S, 32) * 0.5).bfloat16()
        g_h = (torch.randn(S, H, S, 32) * 1.5).bfloat16()
        w_h = (torch.randn(C, C) / C ** 0.5).bfloat16()
        z_h = torch.randn(1, S, S, C).bfloat16()
        od, gd, wd = (ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
                      for t in (o_h, g_h, w_h))
        upd = (o_h.double() * torch.sigmoid(g_h.double())).permute(0, 2, 1, 3).reshape(S, S, C) @ w_h.double()
        ref = z_h.double().reshape(S, S, C) + upd
        kw = dict(S=S, seed=seed)
        for mode, k in CFG.items():
            gt = ttnn.multiply(od, gd, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            u = TQ.out_proj(gt, wd, k, ttnn.bfloat16); ttnn.deallocate(gt)
            count(ttnn.to_torch(u), upd, f"base_{mode}_nores", **kw)
            zz = ttnn.from_torch(z_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            ttnn.add_(zz, ttnn.reshape(u, zz.shape)); ttnn.deallocate(u)
            count(ttnn.to_torch(zz), ref, f"base_{mode}", **kw); ttnn.deallocate(zz)
        for kb in (0, 1):
            TQ.TAIL_KB1 = kb
            u = TQ.gated_out_proj(od, gd, wd, CFG["normal"])
            assert u is not None, "gated_out_proj declined"
            count(ttnn.to_torch(u), upd, f"fused_kb{kb}_nores", **kw); ttnn.deallocate(u)
            zz = ttnn.from_torch(z_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            assert TQ.gated_out_proj(od, gd, wd, CFG["normal"], resid=zz) is zz
            count(ttnn.to_torch(zz), ref, f"fused_kb{kb}", **kw)
            if seed == 0:   # ms per call, residual form, back to back after the warm call above
                ttnn.synchronize_device(dev); t0 = time.perf_counter()
                for _ in range(10):
                    TQ.gated_out_proj(od, gd, wd, CFG["normal"], resid=zz)
                ttnn.synchronize_device(dev)
                log(ev="time", arm=f"fused_kb{kb}", S=S, ms=1e3 * (time.perf_counter() - t0) / 10,
                    aiclk=open(f"/sys/class/tenstorrent/tenstorrent!{NODE}/tt_aiclk").read().split()[0]
                    if NODE >= 0 else None)
            ttnn.deallocate(zz)
        TQ.TAIL_KB1 = 0
        for t in (od, gd, wd):
            ttnn.deallocate(t)
log(ev="done")
