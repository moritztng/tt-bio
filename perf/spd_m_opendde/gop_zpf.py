"""The triangle attention tail with a residual (`triatt_qkv.gated_out_proj`, resid into z), OpenDDE's c384 x 12 heads:
the writer reads each unit's z just before writing it (ZPF 0) or prefetches the next unit's z while it writes this one
(ZPF 1). ms/call per arm and torch.equal of the two outputs.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/gop_zpf.py OUT.json [--sizes 512,480,640]
"""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--sizes", default="512,480,640")
ap.add_argument("--ncall", type=int, default=10); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import triatt_qkv as Q

dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
Q.set_tail("1")
H, D, C = 12, 32, 384
res = {"arch": str(dev.arch()), "cells": {}}
up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
for S in [int(s) for s in a.sizes.split(",")]:
    torch.manual_seed(S)
    o = up(torch.randn(S, H, S, D))
    hg = torch.randn(S, H, S, D)
    w = up(torch.randn(H * D, C) / (H * D) ** 0.5)
    hz = torch.randn(1, S, S, C)
    for gdt in ("bf16", "bfp8"):
        g = up(hg, ttnn.bfloat16 if gdt == "bf16" else ttnn.bfloat8_b)
        names = {"triatt_tail"} | ({"triatt_ln_b8g"} if gdt == "bfp8" else set())
        arms, outs = {}, {}
        with T.levers(",".join(sorted(names))):
            for zpf in (0, 1, 0, 1):
                Q.TAIL_ZPF = zpf
                Q._TAIL_CACHE.clear()
                z = up(hz)
                r = Q.gated_out_proj(o, g, w, ckc, resid=z)
                if r is None:
                    arms[f"zpf{zpf}"] = {"declined": True}
                    continue
                outs.setdefault(zpf, ttnn.to_torch(r))
                z.deallocate()
                zs = [up(hz) for _ in range(a.ncall)]
                ms = []
                for _ in range(a.reps):
                    ttnn.synchronize_device(dev); t0 = time.perf_counter()
                    for i in range(a.ncall):
                        Q.gated_out_proj(o, g, w, ckc, resid=zs[i])
                    ttnn.synchronize_device(dev)
                    ms.append((time.perf_counter() - t0) / a.ncall * 1e3)
                for zz in zs:
                    zz.deallocate()
                arms.setdefault(f"zpf{zpf}", {"ms": []})["ms"] += ms
        Q.TAIL_ZPF = 0
        for v in arms.values():
            if "ms" in v:
                v["ms_min"] = min(v["ms"]); v["ms_med"] = sorted(v["ms"])[len(v["ms"]) // 2]
        if 0 in outs and 1 in outs:
            arms["equal"] = bool(torch.equal(outs[0], outs[1]))
        cell = f"{S}/{gdt}/resid"
        res["cells"][cell] = arms
        print(cell, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "ms"})
                                for k, v in arms.items()}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        g.deallocate()
    o.deallocate(); w.deallocate()
json.dump(res, open(a.out, "w"), indent=1)
