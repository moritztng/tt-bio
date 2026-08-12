#!/usr/bin/env python3
"""SCREEN: what does one host round trip of a pair tensor cost, and what would 200 of them cost?

openfold3's diffusion sampler pads its conditioned tensors by going through host every step:

    tt_bio/openfold3_sample_diffusion.py:_pad_pair    th = ttnn.to_torch(x_dev).float(); ... from_torch
    tt_bio/openfold3_sample_diffusion.py:_pad_tokens  same
    tt_bio/openfold3_diffusion_module.py:_pad_atoms   same
    tt_bio/openfold3_diffusion_module.py:_pad_tokens  same

At 512 tokens n_tok_pad == n_token, so every one of those pads is a NO-OP that still pays a full
device->host->device round trip, 200 times. This prices that, and prices the device-side
alternatives (ttnn.clone as a DRAM-roof reference, and doing nothing at all).

Not a fold-level claim. It is a screen: it predicts the size of ONE lever before anything is built.
"""
import argparse, json, os, time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200, help="diffusion steps to project onto")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()

    F32, BF16 = ttnn.float32, ttnn.bfloat16
    # (label, shape, dtype, calls-per-diffusion-step)  -- the four real pad sites at 512 aa.
    CASES = [
        ("zij  [1,512,512,128] fp32  (_pad_pair)",   (1, 512, 512, 128), F32, 1),
        ("zij  [1,512,512,128] bf16",                (1, 512, 512, 128), BF16, 0),
        ("si   [1,512,384] fp32      (_pad_tokens)", (1, 512, 384),      F32, 1),
        ("ai   [1,512,768] fp32      (dm _pad_tokens)", (1, 512, 768),   F32, 1),
        ("cl   [1,4096,128] fp32     (dm _pad_atoms x3)", (1, 4096, 128), F32, 3),
    ]

    def sync():
        ttnn.synchronize_device(dev)

    def timeit(fn, iters):
        sync(); fn(); sync()                       # warm
        ts = []
        for _ in range(iters):
            sync(); t0 = time.perf_counter(); fn(); sync()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2], min(ts), max(ts)

    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "steps": a.steps, "iters": a.iters, "cases": []}

    for label, shape, dt, per_step in CASES:
        h = torch.randn(*shape)
        x = ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
        nbytes = int(torch.tensor(shape).prod()) * (4 if dt == F32 else 2)

        holder = {}

        def rt():
            th = ttnn.to_torch(x).float()
            y = ttnn.from_torch(th, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
            ttnn.deallocate(y)

        def d2h():
            holder["th"] = ttnn.to_torch(x).float()

        def h2d():
            y = ttnn.from_torch(holder["th"], layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
            ttnn.deallocate(y)

        def clone():
            y = ttnn.clone(x)
            ttnn.deallocate(y)

        rt_ms = timeit(rt, a.iters)
        d2h_ms = timeit(d2h, a.iters)
        h2d_ms = timeit(h2d, a.iters)
        cl_ms = timeit(clone, a.iters)
        row = {
            "case": label, "shape": list(shape), "dtype": str(dt).rsplit(".", 1)[-1],
            "MB": round(nbytes / 1e6, 1), "per_step_calls": per_step,
            "roundtrip_ms": round(rt_ms[0] * 1e3, 3),
            "roundtrip_min_max_ms": [round(rt_ms[1] * 1e3, 3), round(rt_ms[2] * 1e3, 3)],
            "to_torch_ms": round(d2h_ms[0] * 1e3, 3),
            "from_torch_ms": round(h2d_ms[0] * 1e3, 3),
            "device_clone_ms": round(cl_ms[0] * 1e3, 3),
            "clone_implied_GBs": round(2 * nbytes / cl_ms[0] / 1e9, 1),
            "roundtrip_implied_GBs": round(2 * nbytes / rt_ms[0] / 1e9, 2),
            "projected_s_over_steps": round(per_step * a.steps * rt_ms[0], 3),
        }
        res["cases"].append(row)
        print(f"{label:46s} {row['MB']:8.1f} MB  rt {row['roundtrip_ms']:9.3f} ms "
              f"(d2h {row['to_torch_ms']:8.3f} + h2d {row['from_torch_ms']:8.3f})  "
              f"clone {row['device_clone_ms']:7.3f} ms = {row['clone_implied_GBs']:6.1f} GB/s  "
              f"-> {row['projected_s_over_steps']:7.3f} s over {a.steps} steps x{per_step}",
              flush=True)
        holder.clear()
        ttnn.deallocate(x)

    res["projected_total_s"] = round(sum(r["projected_s_over_steps"] for r in res["cases"]), 3)
    print(f"\nPROJECTED total host-round-trip cost inside the 200-step rollout: "
          f"{res['projected_total_s']:.3f} s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
