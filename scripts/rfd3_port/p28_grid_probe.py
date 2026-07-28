"""Per-shape upside and bit-exactness of pinning a core grid on the DiT/encoder linears.

Two questions this answers before any loop-level A/B is worth running:

1. `tt_bio.rfd3` does `from .tenstorrent import CORE_GRID_MAIN` at import time, but
   `_configure_active_compute_grid` rewrites that global when the device opens (11x10 ->
   13x10 on a Blackhole p150). So every `core_grid=CORE_GRID_MAIN` in rfd3.py may be
   pinning 110 of the 130 available cores. Printed, not assumed.

2. `BATCH_INVARIANT_GRID = None` leaves ~430 linears per step on ttnn's default program
   config (rfd3.py documents why: a pinned grid regroups the fp32 accumulation and breaks
   batch invariance). For each real per-step shape this times default vs 11x10 vs 13x10 and
   reports whether the pinned result is BITWISE equal to the default one -- so the
   speed-vs-accuracy tradeoff is per-shape evidence rather than one aggregate guess.

Shapes are read off the shipped module dimensions (RFD3DiffusionModule /
LocalTokenTransformer / LocalAtomTransformer), not hand-typed, so this cannot drift from
the call sites the way p19's baseline did (p20 lesson: benchmark the call site).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPS = 20


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[250],
                    help="I (token count); 250 is the 3359-atom fixture")
    ap.add_argument("--atoms", type=int, nargs="+", default=[3359])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    return ap.parse_args()


def time_linear(x, w, ckc, dtype, grid, reps=REPS):
    import ttnn
    out = ttnn.linear(x, w, core_grid=grid, compute_kernel_config=ckc, dtype=dtype)
    ttnn.synchronize_device(x.device())
    t0 = time.perf_counter()
    for _ in range(reps):
        o = ttnn.linear(x, w, core_grid=grid, compute_kernel_config=ckc, dtype=dtype)
        ttnn.deallocate(o)
    ttnn.synchronize_device(x.device())
    dt = (time.perf_counter() - t0) / reps
    return dt * 1000.0, out


def main():
    args = parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import rfd3

    dev = tt.get_device()
    print(json.dumps({
        "device_grid": [int(dev.compute_with_storage_grid_size().x),
                        int(dev.compute_with_storage_grid_size().y)],
        "tenstorrent.CORE_GRID_MAIN": [tt.CORE_GRID_MAIN.x, tt.CORE_GRID_MAIN.y],
        "rfd3.CORE_GRID_MAIN": [rfd3.CORE_GRID_MAIN.x, rfd3.CORE_GRID_MAIN.y],
        "rfd3.BATCH_INVARIANT_GRID": (None if rfd3.BATCH_INVARIANT_GRID is None else
                                      [rfd3.BATCH_INVARIANT_GRID.x, rfd3.BATCH_INVARIANT_GRID.y]),
    }), flush=True)

    ckc = rfd3._default_compute_kernel_config()
    dt = ttnn.bfloat16
    grids = {
        "default": None,
        "g11x10": ttnn.CoreGrid(y=10, x=11),
        "g13x10": ttnn.CoreGrid(y=10, x=13),
    }
    if int(dev.compute_with_storage_grid_size().x) < 13:
        grids.pop("g13x10")

    C_TOKEN = rfd3.LocalTokenTransformer.C_TOKEN      # 768
    C_S = rfd3.LocalTokenTransformer.C_S              # 384
    C_ATOM = rfd3.RFD3DiffusionModule.C_ATOM          # 128
    T_HIDDEN = None  # read from the real weight below

    # The shapes that carry BATCH_INVARIANT_GRID today, named by their call site.
    # (rows, K, N, per-step call count at n_recycle=2)
    def dit_shapes(I):
        return [
            ("dit.q/k/v/g",  I, C_TOKEN, C_TOKEN, 18 * 2 * 4),
            ("dit.o",        I, C_TOKEN, C_TOKEN, 18 * 2),
            ("dit.adaln_gain", I, C_S, C_TOKEN, 18 * 2 * 2),
            ("dit.adaln_bias", I, C_S, C_TOKEN, 18 * 2 * 2),
            ("dit.t_fc1",    I, C_TOKEN, 4 * C_TOKEN, 18 * 2),
            ("dit.t_fc2",    I, C_TOKEN, 4 * C_TOKEN, 18 * 2),
            ("dit.t_fc3",    I, 4 * C_TOKEN, C_TOKEN, 18 * 2),
            ("dit.out_gate", I, C_S, C_TOKEN, 18 * 2 * 2),
        ]

    def enc_shapes(L):
        return [
            ("enc.q/k/v/g",  L, C_ATOM, C_ATOM, 3 * 4),
            ("enc.o",        L, C_ATOM, C_ATOM, 3),
            ("enc.t_fc1",    L, C_ATOM, 2 * C_ATOM, 3),
            ("enc.t_fc3",    L, 2 * C_ATOM, C_ATOM, 3),
        ]

    rows = []
    for B in args.batches:
        cases = []
        for I in args.tokens:
            cases += [(f"I={I}", B, *s) for s in dit_shapes(I)]
        for L in args.atoms:
            cases += [(f"L={L}", B, *s) for s in enc_shapes(L)]
        for size, batch, name, M, K, N, per_step in cases:
            x = ttnn.from_torch(torch.randn(batch, M, K, dtype=torch.bfloat16),
                                layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
            w = ttnn.from_torch(torch.randn(K, N, dtype=torch.bfloat16),
                                layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
            ref = None
            row = {"size": size, "D": batch, "site": name,
                   "M": M, "K": K, "N": N, "calls_per_step": per_step}
            for label, grid in grids.items():
                try:
                    ms, out = time_linear(x, w, ckc, dt, grid)
                except Exception as exc:  # a grid that will not fit L1 for this shape
                    row[label] = None
                    row[label + "_err"] = str(exc)[:120]
                    continue
                row[label] = round(ms, 4)
                if ref is None:
                    ref = out
                else:
                    row[label + "_exact"] = bool(
                        torch.equal(ttnn.to_torch(out).float(), ttnn.to_torch(ref).float()))
                    ttnn.deallocate(out)
            if ref is not None:
                ttnn.deallocate(ref)
            ttnn.deallocate(x)
            ttnn.deallocate(w)
            base = row.get("default")
            for label in grids:
                if label != "default" and row.get(label):
                    row[label + "_x"] = round(base / row[label], 3)
                    row[label + "_saved_ms_step"] = round(
                        (base - row[label]) * per_step, 2)
            rows.append(row)
            print("ROW " + json.dumps(row), flush=True)

    tot = {}
    for row in rows:
        for label in grids:
            if label == "default":
                continue
            key = (row["D"], label)
            if row.get(label + "_saved_ms_step") is not None:
                tot[key] = tot.get(key, 0.0) + row[label + "_saved_ms_step"]
        key = (row["D"], "default_total")
        if row.get("default") is not None:
            tot[key] = tot.get(key, 0.0) + row["default"] * row["calls_per_step"]
    print("TOTALS " + json.dumps({f"D{d}.{lab}": round(v, 2)
                                  for (d, lab), v in sorted(tot.items())}), flush=True)


if __name__ == "__main__":
    main()
