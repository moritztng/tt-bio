#!/usr/bin/env python3
"""Config sweep for the pair-track layer norm, against the shipped `ttnn.layer_norm` call.

The Wormhole BindCraft 2 replay census (.114 out/whreplayall_eeb25e1ca) has `layer_norm` on
[1, 224, 224, 128] bf16 at 203 us a call, 127 GB/s, 1084 calls a round (0.22 s), where the
elementwise generic_ops on the same tensor run at 279 GB/s. This times the shipped call (HiFi4,
fp32 dest, DRAM out) against other compute configs, ttnn's default program-config knobs, and an
L1 result, every arm with rel L2 against a float64 layer norm of the same bf16 input.

Each arm: warm 2, median of 5 reps of `--inner` calls between syncs.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402

WARM, REPS = 2, 5


def timed(dev, fn, inner):
    for _ in range(WARM):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(inner):
            o = fn(); ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / inner)
    o = fn(); ttnn.synchronize_device(dev)
    return statistics.median(ts), o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--sizes", default="224,195")
    a = ap.parse_args()
    dev = tt.get_device()
    ckc_cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
               else ttnn.types.BlackholeComputeKernelConfig)

    def ckc(fid, acc, approx=False):
        return ckc_cls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=approx,
                       fp32_dest_acc_en=acc, packer_l1_acc=True)

    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,  # noqa: E731
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    prod = ckc("HiFi4", True)
    configs = [("hifi4-fp32", prod), ("hifi4-fp32-approx", ckc("HiFi4", True, True)),
               ("hifi2-fp32", ckc("HiFi2", True)), ("lofi-fp32", ckc("LoFi", True)),
               ("hifi4-bf16", ckc("HiFi4", False)), ("hifi2-bf16", ckc("HiFi2", False))]
    pcs = []
    for kw in ({"use_welford": True}, {"legacy_reduction": False, "legacy_rsqrt": False},
               {"legacy_reduction": False, "legacy_rsqrt": False, "use_welford": True}):
        try:
            pcs.append(("pc_" + "_".join(f"{k}={int(v)}" for k, v in kw.items()),
                        ttnn.LayerNormDefaultProgramConfig(**kw)))
        except Exception as e:                                                  # noqa: BLE001
            print("no program config", kw, e, flush=True)
    res = {"arch": str(dev.arch()), "loadavg": open("/proc/loadavg").read().split()[:3],
           "inner": a.inner, "cells": []}
    torch.manual_seed(0)
    for S in map(int, a.sizes.split(",")):
        C = 128
        xt = torch.randn(1, S, S, C, dtype=torch.bfloat16) * 2 + 0.5
        wt = (1 + 0.1 * torch.randn(C)).to(torch.bfloat16)
        bt = (0.1 * torch.randn(C)).to(torch.bfloat16)
        x64 = xt.double()
        ref64 = torch.nn.functional.layer_norm(x64, (C,), wt.double(), bt.double(), eps=1e-5)
        x, w, b = up(xt), up(wt.reshape(1, C)), up(bt.reshape(1, C))
        nbytes = 2 * xt.numel() * 2
        ln = lambda c, **kw: (lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5,  # noqa: E731
                                                       compute_kernel_config=c, **kw))
        arms = [(n, ln(c)) for n, c in configs]
        arms += [(n, ln(prod, program_config=pc)) for n, pc in pcs]
        arms += [("l1_out", ln(prod, memory_config=ttnn.L1_MEMORY_CONFIG))]
        cell = {"S": S, "C": C, "bytes": nbytes, "arms": []}
        res["cells"].append(cell)
        for name, fn in arms:
            row = {"arm": name}
            try:
                s, o = timed(dev, fn, a.inner)
                got = ttnn.to_torch(o).reshape(ref64.shape); ttnn.deallocate(o)
                d = got.double() - ref64
                row.update(us=1e6 * s, gbs=nbytes / s / 1e9, rel_l2_f64=float(d.norm() / ref64.norm()),
                           finite=bool(torch.isfinite(got).all()))
            except Exception as e:                                              # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            cell["arms"].append(row)
            print(S, row, flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
        for t in (x, w, b):
            ttnn.deallocate(t)
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
