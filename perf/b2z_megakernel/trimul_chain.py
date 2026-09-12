#!/usr/bin/env python3
"""Per-op device attribution of Boltz-2's TriangleMultiplication at one sequence length.

The incumbent this workstream has to beat is the chain as shipped, not a re-derivation of it, so
nothing here re-implements the op sequence: it wraps the ttnn entry points the module calls and
lets the module run. Each wrapper drains the device on both sides of the call, so a row is that
op's own device time plus one drain; the un-instrumented whole-call time is reported next to the
sum so the drain tax is visible rather than hidden.

Bytes are counted the same way `perf/b2x_fusion_boundary/block_attrib.py` counts them -- inputs
read once, outputs written once, a fused op once -- so the two tables are comparable, and are
reported in Z, one Z being the bf16 pair tensor at this size.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn
from tt_bio import tenstorrent as TT
from tt_bio import mm_dualnoc, reblock_permute, trimul_tail

# Ops that read only part of their input. `reblock_permute_gated` takes the p and g slices of a
# four-role fused projection, so it reads half of what it is handed; counting the whole tensor
# would price the trimul at 25 Z instead of the 20 Z it moves.
READ_FRAC = {"E6:reblock_permute_gated": 0.5}

REC: list[dict] = []
DEPTH = [0]
ON = [False]


def _shape(t):
    try:
        return "x".join(str(int(d)) for d in t.shape)
    except Exception:                                                     # noqa: BLE001
        return "?"


def _where(t):
    """DRAM / L1 / '?' for a tensor. The per-op GB/s column is only a DRAM number for the
    operands that actually live in DRAM; `_pair_proj_linear(l1_out=True)` and the trimul tail
    keep theirs in L1 at c_z = 128, and counting those as DRAM traffic overstates the roof
    denominator for exactly the ops F1 already found to be L1-resident."""
    try:
        return "L1" if t.memory_config().buffer_type == ttnn.BufferType.L1 else "DRAM"
    except Exception:                                                     # noqa: BLE001
        return "?"


def _dram_bytes(t):
    return _bytes(t) if _where(t) == "DRAM" else 0


def _bytes(t):
    try:
        n = 1
        for d in t.shape:
            n *= int(d)
        return n * (2 if t.dtype in (ttnn.bfloat16,) else 4 if t.dtype == ttnn.float32 else 1)
    except Exception:                                                     # noqa: BLE001
        return 0


def _wrap(obj, name, label=None):
    fn = getattr(obj, name)
    label = label or name

    def inner(*a, **k):
        if not ON[0] or DEPTH[0]:
            return fn(*a, **k)
        DEPTH[0] += 1
        try:
            ttnn.synchronize_device(TT.get_device())
            t0 = time.perf_counter()
            out = fn(*a, **k)
            ttnn.synchronize_device(TT.get_device())
            dt = time.perf_counter() - t0
        finally:
            DEPTH[0] -= 1
        ins = [x for x in a if isinstance(x, ttnn.Tensor)]
        outs = [out] if isinstance(out, ttnn.Tensor) else [
            x for x in (out or ()) if isinstance(x, ttnn.Tensor)]
        frac = READ_FRAC.get(label, 1.0)
        rb = sum(_bytes(x) for x in ins) * frac
        wb = sum(_bytes(x) for x in outs)
        # the same two sums restricted to operands that really live in DRAM
        rb_d = sum(_dram_bytes(x) for x in ins) * frac
        wb_d = sum(_dram_bytes(x) for x in outs)
        REC.append({"op": label, "ms": dt * 1e3, "in": [_shape(x) for x in ins],
                    "out": _shape(outs[0]) if outs else "",
                    "in_mem": [_where(x) for x in ins],
                    "out_mem": [_where(x) for x in outs],
                    "read_B": rb, "write_B": wb,
                    "read_dram_B": rb_d, "write_dram_B": wb_d})
        return out

    setattr(obj, name, inner)


def install():
    for n in ("layer_norm", "matmul", "multiply_", "multiply", "permute", "transpose",
              "chunk", "clone", "reallocate", "slice", "concat", "typecast", "add", "add_"):
        if hasattr(ttnn, n):
            _wrap(ttnn, n)
    _wrap(ttnn.experimental, "minimal_matmul")
    for n in ("reblock_permute", "reblock_permute_back", "reblock_permute_gated"):
        _wrap(reblock_permute, n, "E6:" + n)
    _wrap(mm_dualnoc, "in_proj", "dualnoc:in_proj")
    _wrap(trimul_tail, "fused_tail", "F1:fused_tail")
    _wrap(TT, "_pair_proj_linear", "_pair_proj_linear")


def build(n, c_z, device, seed=0):
    g = torch.Generator().manual_seed(seed)

    def rnd(*s):
        return torch.randn(*s, generator=g) * 0.05

    sd = {
        "norm_in.weight": torch.ones(c_z), "norm_in.bias": torch.zeros(c_z),
        "norm_out.weight": torch.ones(c_z), "norm_out.bias": torch.zeros(c_z),
        "p_in.weight": rnd(2 * c_z, c_z), "g_in.weight": rnd(2 * c_z, c_z),
        "p_out.weight": rnd(c_z, c_z), "g_out.weight": rnd(c_z, c_z),
    }
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if device.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)
    mods = [TT.TriangleMultiplication(ending=e, state_dict=sd, compute_kernel_config=ckc)
            for e in (False, True)]
    for m in mods:
        m.prewarm(n, 1)
    z = ttnn.from_torch(rnd(1, n, n, c_z), layout=ttnn.TILE_LAYOUT, device=device,
                        dtype=ttnn.bfloat16)
    mask = ttnn.from_torch(torch.ones(1, n, n), layout=ttnn.TILE_LAYOUT, device=device,
                           dtype=ttnn.bfloat16)
    return mods, z, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c-z", type=int, default=128)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = TT.get_device()
    grid = device.compute_with_storage_grid_size()
    n_cores = grid.x * grid.y
    l1 = ttnn.get_max_worker_l1_unreserved_size()
    arch = str(device.arch())

    install()
    mods, z, mask = build(a.n, a.c_z, device)
    Z = a.n * a.n * a.c_z * 2

    # warm: first call compiles programs
    for m in mods:
        for _ in range(2):
            ttnn.deallocate(m(z, mask))
    ttnn.synchronize_device(device)

    walls = {}
    for name, m in (("start", mods[0]), ("end", mods[1])):
        ts = []
        for _ in range(a.iters):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = m(z, mask)
            ttnn.synchronize_device(device)
            ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(out)
        walls[name] = {"median_ms": statistics.median(ts), "min_ms": min(ts), "runs": ts}

    per_op = {}
    for name, m in (("start", mods[0]), ("end", mods[1])):
        REC.clear()
        ON[0] = True
        ttnn.deallocate(m(z, mask))
        ON[0] = False
        per_op[name] = list(REC)

    res = {
        "_what": "per-op device attribution of Boltz-2 TriangleMultiplication, shipped path",
        "arch": arch, "n_cores": n_cores, "grid": f"{grid.x}x{grid.y}",
        "l1_unreserved_per_core_B": l1,
        "n": a.n, "c_z": a.c_z, "Z_bytes": Z, "iters": a.iters,
        "wall": walls, "per_op": per_op,
        "sum_instrumented_ms": {k: sum(r["ms"] for r in v) for k, v in per_op.items()},
        "sum_Z": {k: sum(r["read_B"] + r["write_B"] for r in v) / Z for k, v in per_op.items()},
        # the same total counting ONLY operands resident in DRAM -- the denominator a streaming
        # roof is actually entitled to
        "sum_Z_dram": {k: sum(r["read_dram_B"] + r["write_dram_B"] for r in v) / Z
                       for k, v in per_op.items()},
    }
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt[:400])
    print(f"\n== {arch} {grid.x}x{grid.y}={n_cores} cores, L1 unreserved/core = {l1} B ==")
    for k, v in walls.items():
        print(f"trimul {k:5s}: median {v['median_ms']:8.3f} ms  min {v['min_ms']:8.3f} ms   "
              f"(sum of instrumented ops {res['sum_instrumented_ms'][k]:8.3f} ms, "
              f"{res['sum_Z'][k]:.1f} Z of which {res['sum_Z_dram'][k]:.1f} Z in DRAM)")
    for k, v in per_op.items():
        print(f"\n-- {k} --")
        for r in v:
            zt = (r["read_B"] + r["write_B"]) / Z
            zd = (r["read_dram_B"] + r["write_dram_B"]) / Z
            bw = (r["read_dram_B"] + r["write_dram_B"]) / max(r["ms"] - 0.036, 1e-9) / 1e6
            print(f"  {r['ms']:8.3f} ms  {zt:5.2f} Z {zd:5.2f} D {bw:7.1f} GB/s  "
                  f"{r['op']:28s} {'|'.join(r['in_mem']):14s} -> "
                  f"{'|'.join(r['out_mem']):5s} {'|'.join(r['in']):34s} -> {r['out']}")


if __name__ == "__main__":
    main()
