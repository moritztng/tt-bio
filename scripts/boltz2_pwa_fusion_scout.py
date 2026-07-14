"""Boltz-2 PairWeightedAveraging fusion scout.

Profiles the MSA module's PairWeightedAveraging primitive with REAL Boltz-2
checkpoint weights (4 blocks, n_heads=8, head_dim=32) at Boltz-2-representative
shapes (MSA padded to 1024, seq 128/256/512). Measures the dispatch-collapse
ceiling via ttnn trace replay (the per-head Python loop is a candidate
dispatch pile) plus per-op-sync accounting and a bit-exact parity check.

PWA is shared between Boltz-2 and Protenix-v2 (same tt_bio.tenstorrent class,
same 8-head per-call Python loop); this run uses Boltz-2's own weights/dims.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import torch
import ttnn


def _compare(a: torch.Tensor, b: torch.Tensor, chunk: int = 1 << 22) -> dict:
    x, y = a.reshape(-1).double(), b.reshape(-1).double()
    n = x.numel()
    sx = sy = sxx = syy = sxy = 0.0
    max_abs = 0.0
    finite = True
    for s in range(0, n, chunk):
        xd, yd = x[s:s + chunk], y[s:s + chunk]
        finite = finite and bool(torch.isfinite(xd).all() and torch.isfinite(yd).all())
        max_abs = max(max_abs, float((xd - yd).abs().max()))
        sx += float(xd.sum()); sy += float(yd.sum())
        sxx += float((xd * xd).sum()); syy += float((yd * yd).sum())
        sxy += float((xd * yd).sum())
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    return {"pcc": cov / max((vx * vy) ** 0.5, 1e-30), "max_abs": max_abs, "finite": finite}


def _build_pwa_boltz2(state, config):
    from tt_bio.tenstorrent import PairWeightedAveraging
    modules = []
    for index in range(4):
        prefix = f"msa_module.layers.{index}.pair_weighted_averaging."
        weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not weights:
            continue
        n_heads = int(weights["proj_z.weight"].shape[0])  # (8, 128)
        head_dim = int(weights["proj_m.weight"].shape[0]) // n_heads  # 256/8 = 32
        modules.append(PairWeightedAveraging(head_dim, n_heads, weights, config))
    return modules


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--msa-depth", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--checkpoint", default="/home/ttuser/.boltz/boltz2_conf.ckpt")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20260712)

    from tt_bio import tenstorrent as T
    T.set_fast_mode(False)
    device = T.get_device(1 << 30)  # trace region
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    print("loading Boltz-2 checkpoint:", args.checkpoint, flush=True)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    sd = ck["state_dict"] if "state_dict" in ck else ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    modules = _build_pwa_boltz2(sd, config)
    if not modules:
        raise RuntimeError("no Boltz-2 PWA modules found in checkpoint")
    c_m = int(modules[0].weights["proj_m.weight"].shape[1])
    c_z = int(modules[0].weights["proj_z.weight"].shape[1])
    print(f"built {len(modules)} PWA blocks: heads={modules[0].n_heads} "
          f"head_dim={modules[0].head_dim} c_m={c_m} c_z={c_z}", flush=True)

    def run_eager(m, z, capture=False):
        saved = {}
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for i, mod in enumerate(modules):
            out = mod(m, z)
            if capture and i in (0, len(modules) - 1):
                ttnn.synchronize_device(device)
                saved[i] = torch.Tensor(ttnn.to_torch(out)).float()
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        return time.perf_counter() - t0, saved

    def run_trace(m, z):
        outs = []
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outs = [mod(m, z) for mod in modules]
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        samples = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            samples.append(time.perf_counter() - t0)
        saved = {i: torch.Tensor(ttnn.to_torch(o)).float() for i, o in enumerate(outs)}
        ttnn.release_trace(device, tid)
        for o in outs:
            try: ttnn.deallocate(o)
            except Exception: pass
        return statistics.median(samples), samples, saved

    for size in args.sizes:
        g = torch.Generator().manual_seed(20260712 + size)
        gz = torch.Generator().manual_seed(99 + size)
        m_host = torch.randn((1, args.msa_depth, size, c_m), generator=g, dtype=torch.bfloat16)
        z_host = torch.randn((1, size, size, c_z), generator=gz, dtype=torch.bfloat16)
        m = ttnn.from_torch(m_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        z = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        print(f"warming {len(modules)} Boltz-2 PWA blocks N={size} MSA={args.msa_depth}", flush=True)
        run_eager(m, z)
        base_samples = [run_eager(m, z)[0] for _ in range(args.repeats)]
        _, base_saved = run_eager(m, z, capture=True)
        trace_s, trace_samples, trace_saved = run_trace(m, z)
        base_s = statistics.median(base_samples)
        rec = {
            "component": "boltz2_pair_weighted_averaging",
            "N": size, "msa_depth": args.msa_depth, "blocks": len(modules),
            "heads": modules[0].n_heads, "head_dim": modules[0].head_dim,
            "baseline_s": base_s, "baseline_samples_s": base_samples,
            "trace_floor_s": trace_s, "trace_floor_samples_s": trace_samples,
            "trace_speedup": base_s / trace_s,
            "trace_parity": {str(i): _compare(base_saved[i], trace_saved[i]) for i in base_saved},
        }
        print(json.dumps(rec, sort_keys=True), flush=True)
        ttnn.deallocate(m); ttnn.deallocate(z)
        del m_host, z_host, base_saved, trace_saved
        gc.collect()


if __name__ == "__main__":
    main()
