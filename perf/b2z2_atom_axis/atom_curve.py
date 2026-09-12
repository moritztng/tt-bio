#!/usr/bin/env python3
"""Fit the atom track's own cost curve, t = a + b * n_atoms, before anything is built.

`b2z2-sharded-sampler` killed the token-axis sampler shard by fitting the token DiT:
t = 8.9747 ms + 0.013579 ms/row, so 56.3 % of it does not depend on the token count and a
perfect free shard tops out at 1.2825x. That verdict is about the TOKEN axis. This measures the
same thing on the ATOM axis, which is the axis the atom encoder and decoder are extensive in.

Instrument: a real atom-level `tt_bio.tenstorrent.DiffusionTransformer` (3 layers, dim 128,
4 heads, atom_level=True) at production geometry, built from a torch-reference state_dict, fed
the production window layout, captured as a ttnn trace and replayed -- the same execution mode
the shipped fold uses for the diffusion step (`DiffusionModule._capture_diff_trace`). Weights
are seeded random, which is honest for a timing screen: bf16 matmul time does not depend on the
values, and every arm compared later shares one weight set.

Widths are window counts NW; n_atoms = 32 * NW. The production 512 aa fold runs NW = 140
(4480 atoms), and that point is checked against the committed census (encoder 6.324 ms,
decoder 6.016 ms per step, whglx/WH) as the instrument check.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

W = 32          # ATOM_WINDOW: queries per window
H = 128         # ATOM_DIM: keys per window
DIM = 128       # atom_s
HEADS = 4       # ATOM_N_HEADS
DEPTH = 3       # ATOM_N_LAYERS


def make_weights(torch, seed: int):
    """Atom-transformer weights with the reference's own shapes and key names.

    `init.final_init_` zeroes the two output projections, which would make the block's output
    independent of its attention and blind any later parity control. Zero tensors are refilled
    with small noise; everything else is the reference's own init.
    """
    from tt_bio.boltz2 import DiffusionTransformer as TorchDT

    torch.manual_seed(seed)
    m = TorchDT(depth=DEPTH, heads=HEADS, dim=DIM, dim_single_cond=DIM)
    sd = {}
    g = torch.Generator().manual_seed(seed + 1)
    for k, v in m.state_dict().items():
        v = v.detach().float().clone()
        if v.abs().max() == 0:
            v = torch.randn(v.shape, generator=g) * 0.02
        sd[k] = v
    return sd


def build_inputs(torch, NW: int, seed: int):
    """Host tensors in the exact layout `Diffusion.__call__` hands the atom transformer."""
    from tt_bio.boltz2 import get_indexing_matrix

    g = torch.Generator().manual_seed(seed)
    a = torch.randn(1, NW, W, DIM, generator=g)
    s = torch.randn(1, NW, W, DIM, generator=g)
    ki = get_indexing_matrix(NW, W, H, torch.device("cpu"))       # (2NW, 8NW) one-hot

    # The production additive mask: key slots whose source half-window falls outside the atom
    # axis gather nothing, and `_populate_diffusion_cache` turns those into -1e9 in the bias.
    ones = torch.ones(1, 2 * NW, W // 2, 1)
    valid = torch.einsum("bjid,jk->bkid", ones, ki).reshape(1, NW, H, 1)   # single_to_keys
    mask = (1.0 - valid).reshape(NW, 1, 1, H) * -1e9

    z = []
    for _ in range(DEPTH):
        b = torch.randn(NW, HEADS, W, H, generator=g) * 0.1
        z.append(((b + mask) * (W ** 0.5)).contiguous())
    return a, s, z, ki


def to_device(ttnn, dev, a, s, z, ki):
    f = lambda x, dt=None: ttnn.from_torch(
        x, device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt or ttnn.bfloat16)
    return f(a), f(s), [f(x) for x in z], f(ki, ttnn.bfloat4_b)


def timed_trace(ttnn, dev, fn, reps: int, blocks: int):
    """Capture `fn` once, replay it, return per-block median ms of one replay."""
    fn(); fn()                                   # compile + populate the lazy per-layer caches
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ms = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ms.append(1e3 * (time.perf_counter() - t0) / reps)
    ttnn.release_trace(dev, tid)
    return ms, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows", default="28,56,84,112,140",
                    help="window counts NW; n_atoms = 32*NW")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    widths = [int(x) for x in args.windows.split(",")]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    assert Path(T.__file__).resolve().parents[1] == ROOT, \
        f"importing tt_bio from {T.__file__}, not {ROOT}"
    assert (W, H, DIM, HEADS, DEPTH) == (
        T.ATOM_WINDOW, T.ATOM_DIM, T.ATOM_DIM, T.ATOM_N_HEADS, T.ATOM_N_LAYERS)

    dev = T.get_device()
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)

    out = {"env": {"host": os.uname().nodename,
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "arch": T.arch_name(), "grid": str(T.CORE_GRID_MAIN),
                   "trace_region": T.trace_region_size(),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "loadavg_start": open("/proc/loadavg").read().split()[:3],
                   "reps": args.reps, "blocks": args.blocks, "seed": args.seed},
           "rows": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sd = make_weights(torch, args.seed)
    for part in ("encoder", "decoder"):
        # Two independent 3-layer stacks in the fold; same shape, same weights here. Built
        # fresh per width so no lazy per-layer cache crosses a width boundary.
        for NW in widths:
            a_pt, s_pt, z_pt, ki_pt = build_inputs(torch, NW, args.seed + 7)
            mod = T.DiffusionTransformer(n_layers=DEPTH, dim=DIM, n_heads=HEADS,
                                         atom_level=True, state_dict=sd,
                                         compute_kernel_config=ckc)
            a, s, z, ki = to_device(ttnn, dev, a_pt, s_pt, z_pt, ki_pt)
            ms, _ = timed_trace(ttnn, dev, lambda: mod(a, s, z, ki), args.reps, args.blocks)
            row = {"part": part, "NW": NW, "n_atoms": W * NW,
                   "ms": round(st.median(ms), 5),
                   "spread_pct": round(100 * (max(ms) - min(ms)) / st.median(ms), 3),
                   "blocks_ms": [round(x, 5) for x in ms],
                   "loadavg": open("/proc/loadavg").read().split()[0]}
            out["rows"].append(row)
            print(f"  {part:8s} NW={NW:4d} atoms={W*NW:5d} {row['ms']:9.4f} ms "
                  f"(spread {row['spread_pct']:.2f} %, load {row['loadavg']})", flush=True)
            args.out.write_text(json.dumps(out, indent=1))
            for t in (a, s, ki, *z):
                ttnn.deallocate(t)
            del mod

    # least-squares fit t = a + b * n_atoms, per part and pooled
    fits = {}
    for part in ("encoder", "decoder", "both"):
        rows = [r for r in out["rows"] if part in (r["part"], "both")]
        if part == "both":
            by_n = {}
            for r in out["rows"]:
                by_n.setdefault(r["n_atoms"], 0.0)
                by_n[r["n_atoms"]] += r["ms"]
            pts = sorted(by_n.items())
        else:
            pts = sorted((r["n_atoms"], r["ms"]) for r in rows)
        n = len(pts)
        if n < 2:
            continue
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts); sxy = sum(p[0] * p[1] for p in pts)
        b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        a0 = (sy - b * sx) / n
        ybar = sy / n
        ss_res = sum((y - (a0 + b * x)) ** 2 for x, y in pts)
        ss_tot = sum((y - ybar) ** 2 for x, y in pts)
        x_prod = W * max(widths)
        t_prod = a0 + b * x_prod
        fits[part] = {
            "a_ms": round(a0, 5), "b_ms_per_atom": round(b, 8),
            "r2": round(1 - ss_res / ss_tot, 6),
            "resid_pct": [round(100 * (y - (a0 + b * x)) / y, 3) for x, y in pts],
            "t_at_production_ms": round(t_prod, 5),
            "constant_fraction_pct": round(100 * a0 / t_prod, 3),
            "perfect_2chip_ratio": round(t_prod / (a0 + b * x_prod / 2), 5),
        }
    out["fit"] = fits
    out["env"]["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(fits, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
