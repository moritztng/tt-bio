#!/usr/bin/env python3
"""P4 / p2-layout-kernel -- the mechanism tests.

M1  transaction-count vs bytes.  A tile-crossing channel move issues one face-row transaction per
    (source tile, destination tile) pair: 32 per tile at C=32, INDEPENDENT of the element size.
    Going bf16 -> fp32 doubles the bytes per transaction and per tile while holding the transaction
    COUNT fixed. If the move is transaction-count-bound its us/tile is flat; if it is byte-bound it
    doubles. The clone on the same tensors is the control: it must double either way.

M2  core utilisation, measured.  The prototype splits work as num_groups = Nt*Nt over the 110-core
    grid, so N=320 -> 100 groups -> 100 of 110 cores, one group each, and N=352 -> 121 groups ->
    110 cores with 11 of them carrying two. If the split is what it says it is, per-tile cost jumps
    ~2x across that boundary and is flat inside it.

M3  shape ladder.  Is the per-tile cost of the baseline and of the prototype flat in N?
"""
import argparse, json, time
from pathlib import Path

import torch
import ttnn


def timeit(device, fn, reps=10, warmup=3):
    for _ in range(warmup):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    ts.sort()
    return sum(ts[: max(1, len(ts) // 2)]) / max(1, len(ts) // 2)


L1 = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="perf/p2_layout_kernel/tileface_mech.json")
    a = ap.parse_args()
    global L1
    device = ttnn.open_device(device_id=0)
    L1 = ttnn.L1_MEMORY_CONFIG
    g = device.compute_with_storage_grid_size()
    ncores = g.x * g.y
    has_reblock = hasattr(ttnn.experimental, "reblock_permute")
    R = {"cores": ncores, "has_reblock": has_reblock, "M1": {}, "M2": {}, "M3": {}}
    torch.manual_seed(0)

    def dev(t, dt):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dt, memory_config=L1)

    # ------------------------------------------------------------------ M1 transaction count vs bytes
    print("== M1: bf16 -> fp32 doubles BYTES per transaction, holds transaction COUNT fixed ==")
    N, C = 320, 32
    nt = N * N * C // 1024
    x = torch.randn(1, N, N, C)
    for dt, name in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
        try:
            t = dev(x, dt)
            cl = timeit(device, lambda t=t: ttnn.clone(t, memory_config=L1), a.reps)
            pm = timeit(device, lambda t=t: ttnn.permute(t, (0, 3, 1, 2), memory_config=L1), a.reps)
            R["M1"][name] = {"clone_us": round(cl, 2), "permute_us": round(pm, 2),
                             "clone_us_per_tile_per_core": round(cl * ncores / nt, 4),
                             "permute_us_per_tile_per_core": round(pm * ncores / nt, 4),
                             "reorder_excess_us_per_tile_per_core": round((pm - cl) * ncores / nt, 4)}
            print(f"  {name}: clone {cl:8.2f} us ({cl*ncores/nt:.4f} us/tile/core)   "
                  f"permute(0,3,1,2) {pm:8.2f} us ({pm*ncores/nt:.4f} us/tile/core)   "
                  f"reorder excess {(pm-cl)*ncores/nt:.4f} us/tile/core")
            ttnn.deallocate(t)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED {type(e).__name__}: {e}")
            R["M1"][name] = {"error": str(e)[:200]}

    # ---------------------------------------------------------------------- M2/M3 shape + core ladder
    print("\n== M2/M3: shape ladder (num_groups = Nt*Nt against a 110-core grid) ==")
    for N in (128, 192, 320, 352, 384):
        nt = N * N * C // 1024
        groups = (N // 32) ** 2
        cores_used = min(groups, ncores)
        per_core_groups = -(-groups // ncores)
        xi = torch.randn(1, N, N, C)
        xo = torch.randn(1, C, N, N)
        ai, ao = dev(xi, ttnn.bfloat16), dev(xo, ttnn.bfloat16)
        row = {"tiles": nt, "groups": groups, "cores_used": cores_used,
               "groups_per_busiest_core": per_core_groups}
        cl = timeit(device, lambda ai=ai: ttnn.clone(ai, memory_config=L1), a.reps)
        pi = timeit(device, lambda ai=ai: ttnn.permute(ai, (0, 3, 1, 2), memory_config=L1), a.reps)
        po = timeit(device, lambda ao=ao: ttnn.permute(ao, (0, 2, 3, 1), memory_config=L1), a.reps)
        row.update(clone_us=round(cl, 2), perm_in_us=round(pi, 2), perm_out_us=round(po, 2))
        # per tile per ENGAGED core, which is the unit the org compares in
        f = lambda us: round(us * cores_used / nt, 4)  # noqa: E731
        row.update(clone_uptc=f(cl), perm_in_uptc=f(pi), perm_out_uptc=f(po))
        if has_reblock:
            ri = timeit(device, lambda ai=ai: ttnn.experimental.reblock_permute(ai, memory_config=L1), a.reps)
            ro = timeit(device, lambda ao=ao: ttnn.experimental.reblock_permute_back(ao, memory_config=L1), a.reps)
            eq_i = bool(torch.equal(ttnn.to_torch(ttnn.experimental.reblock_permute(ai, memory_config=L1)),
                                    ttnn.to_torch(ttnn.permute(ai, (0, 3, 1, 2), memory_config=L1))))
            eq_o = bool(torch.equal(ttnn.to_torch(ttnn.experimental.reblock_permute_back(ao, memory_config=L1)),
                                    ttnn.to_torch(ttnn.permute(ao, (0, 2, 3, 1), memory_config=L1))))
            row.update(reblock_in_us=round(ri, 2), reblock_out_us=round(ro, 2),
                       reblock_in_uptc=f(ri), reblock_out_uptc=f(ro),
                       torch_equal_in=eq_i, torch_equal_out=eq_o,
                       speedup_in=round(pi / ri, 3), speedup_out=round(po / ro, 3))
        R["M3"][str(N)] = row
        print(f"  N={N:4d} tiles={nt:5d} groups={groups:4d} cores={cores_used:3d} "
              f"(busiest core {per_core_groups} group(s))  clone {f(cl):.3f}  "
              f"perm_in {f(pi):.3f}  perm_out {f(po):.3f}"
              + (f"  reblock_in {f(ri):.3f} ({pi/ri:.2f}x, eq={eq_i})"
                 f"  reblock_out {f(ro):.3f} ({po/ro:.2f}x, eq={eq_o})" if has_reblock else "")
              + "   us/tile/engaged-core")
        ttnn.deallocate(ai)
        ttnn.deallocate(ao)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=2))
    print("\nwrote", a.out)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
