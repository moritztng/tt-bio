"""The triangle attention tail (`triatt_qkv.gated_out_proj`) with and without TAIL_STAGGER, OpenDDE's c384 x 12 heads:
ms/call per arm, and torch.equal of the output against the unstaggered arm.

cs11 (9LLG 512, WH, 1000 MHz): the starting block's tail (resid into z) reads 6.15 ms/call at 0.53 of roof in normal,
3.95 in fast (bfp8 g); the ending block's (no resid) 3.69 / 3.59. Hypothesis: a unit's 12 head tiles are St pages
apart, so at St 16 they hit 3 of 12 DRAM banks, and 56 of 72 cores start on an even row tile.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/gop_stagger.py OUT.json [--sizes 512,480,640,608]
"""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--sizes", default="512,480,640,608,448")
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
dg = dev.dram_grid_size()
res = {"arch": str(dev.arch()), "dram_grid": [int(dg.x), int(dg.y)], "cells": {}}
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
        for resid in (True, False):
            cell = f"{S}/{gdt}/{'resid' if resid else 'out'}"
            names = {"triatt_tail"} | ({"triatt_ln_b8g"} if gdt == "bfp8" else set())
            arms, outs = {}, {}
            with T.levers(",".join(sorted(names))):
                for stg in (False, True, False, True):
                    Q.TAIL_STAGGER = stg
                    Q._TAIL_CACHE.clear()
                    name = "stagger" if stg else "ship"

                    def call():
                        z = up(hz) if resid else None
                        r = Q.gated_out_proj(o, g, w, ckc, resid=z)
                        return r, z
                    r, z = call()
                    if r is None:
                        arms[name] = {"declined": True}
                        continue
                    outs.setdefault(name, ttnn.to_torch(r))
                    r.deallocate()
                    # time with a preallocated z per call (in place), as the fold does
                    zs = [up(hz) for _ in range(a.ncall)] if resid else [None] * a.ncall
                    ms = []
                    for _ in range(a.reps):
                        ttnn.synchronize_device(dev); t0 = time.perf_counter()
                        for i in range(a.ncall):
                            y = Q.gated_out_proj(o, g, w, ckc, resid=zs[i])
                            if not resid:
                                y.deallocate()
                        ttnn.synchronize_device(dev)
                        ms.append((time.perf_counter() - t0) / a.ncall * 1e3)
                    for zz in zs:
                        if zz is not None:
                            zz.deallocate()
                    arms.setdefault(name, {"ms": []})["ms"] += ms
            for v in arms.values():
                if "ms" in v:
                    v["ms_min"] = min(v["ms"]); v["ms_med"] = sorted(v["ms"])[len(v["ms"]) // 2]
            if "ship" in outs and "stagger" in outs:
                arms["equal"] = bool(torch.equal(outs["ship"], outs["stagger"]))
            res["cells"][cell] = arms
            print(cell, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "ms"})
                                    for k, v in arms.items()}), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        g.deallocate()
    o.deallocate(); w.deallocate()
json.dump(res, open(a.out, "w"), indent=1)
