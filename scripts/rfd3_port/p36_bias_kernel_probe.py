"""L6a: the fused sparse-bias kernel, gated on torch.equal and priced against the shipped chain.

The shipped path builds RFD3's dense attention bias in three ops -- a cached ``-1e4`` bf16
template, ``ttnn.scatter`` of the pair bias into it, and a ``bf16 -> fp32`` typecast -- and
``state/rfd3-host-half.md`` §3 prices those at 5.475 ms/call at the production shape, of
which 4.683 is the scatter's out-of-place copy of 45.14 M elements at a per-element-limited
9.6 G elem/s. ``tt_bio/rfd3_bias.py`` writes the fp32 result directly instead.

Two gates, both required, and the first is the one that matters:

1. ``torch.equal`` on the full [1, H, L, N] fp32 result against the three-op chain, on three
   index distributions -- random-distinct (worst case for how far pokes scatter across the
   tile grid), banded-contiguous (worst case for how many land in one tile: 32 rows x 32
   columns, which is exactly the kernel's per-slot repair bound), and the real trajectory's
   own indices when a dump is available.
2. device time per call, both arms, sync-bracketed, after a warm call.

Run:
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half PYTHONPATH=$PWD \\
      /home/ttuser/tt-bio-dev/env/bin/python3 scripts/rfd3_port/p36_bias_kernel_probe.py \\
      --out perf/p36/bias_kernel.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import ttnn

from tt_bio import rfd3_bias

REPS = 5


def align_tile(n: int) -> int:
    return ((n + 31) // 32) * 32


def make_indices(kind: str, L: int, K: int, seed: int = 0) -> torch.Tensor:
    """[1, L, K] int64, sorted ascending along K, distinct within a row."""
    g = torch.Generator().manual_seed(seed)
    if kind == "random":
        idx = torch.rand(L, L, generator=g).topk(K, dim=-1, largest=False).indices
    elif kind == "banded":
        # Every row's K keys contiguous and centred on the diagonal: a single output tile
        # then takes 32 pokes from each of its 32 rows, the kernel's undo-list worst case.
        base = torch.arange(L).clamp(0, L - K).unsqueeze(1)
        idx = base + torch.arange(K).unsqueeze(0)
    elif kind == "mixed":
        half = K // 2
        base = torch.arange(L).clamp(0, L - half).unsqueeze(1)
        near = base + torch.arange(half).unsqueeze(0)
        far = torch.randint(0, L, (L, K), generator=g)
        idx = torch.cat([near, far], dim=-1)
    else:
        raise SystemExit(f"unknown index kind {kind}")
    # Distinct + sorted, exactly as _create_attention_indices leaves it (it ends in sort).
    out = torch.empty(L, K, dtype=torch.int64)
    for i in range(L):
        u = torch.unique(idx[i])
        if len(u) < K:                                  # top up with unused columns
            free = torch.ones(L, dtype=torch.bool)
            free[u] = False
            u = torch.cat([u, torch.nonzero(free).squeeze(1)[: K - len(u)]]).sort().values
        out[i] = u[:K]
    return out.unsqueeze(0)


def timed(fn, device, reps=REPS):
    fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e3)
        if r is not None:
            ttnn.deallocate(r)
    ts.sort()
    return ts[len(ts) // 2], ts


def one_shape(device, H, L, K, kind, results, slots=(None,)):
    N = align_tile(L)
    print(f"\n=== H{H}_L{L}_K{K}_{kind} ===", flush=True)

    idx = make_indices(kind, L, K)
    pb = torch.randn(1, H, L, K)
    pair_bias = ttnn.from_torch(
        pb, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    idx_tiled = ttnn.from_torch(
        idx.unsqueeze(1).expand(1, H, L, K).contiguous().to(torch.int32),
        layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.uint32,
    )
    idx_rm = ttnn.from_torch(
        idx.unsqueeze(1).to(torch.int32).contiguous(),
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32,
    )
    template = ttnn.full(
        (1, H, L, N), -1e4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    def ref():
        b = ttnn.scatter(template, 3, idx_tiled, pair_bias)
        f = ttnn.typecast(b, ttnn.float32, memory_config=b.memory_config())
        ttnn.deallocate(b)
        return f

    def got():
        return rfd3_bias.sparse_bias_fp32(pair_bias, idx_rm, device=device)

    ms_ref, all_ref = timed(ref, device)
    for s in slots:
        if s is not None:
            rfd3_bias.OUT_SLOTS = s
            rfd3_bias._CACHE.clear()
        tag = f"H{H}_L{L}_K{K}_{kind}" + (f"_slots{s}" if s is not None else "")
        a, b = ref(), got()
        ta, tb = ttnn.to_torch(a), ttnn.to_torch(b)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        equal = bool(torch.equal(ta, tb))
        maxabs = float((ta - tb).abs().max())
        if not equal:
            d = (ta != tb).nonzero()
            print(f"  {len(d)} differing elements of {ta.numel()}; first 5 {d[:5].tolist()}")
            for pos in d[:5].tolist():
                print(f"   at {pos}: ref={ta[tuple(pos)]:.6f} got={tb[tuple(pos)]:.6f}")
        ms_got, all_got = timed(got, device)
        print(f"slots={rfd3_bias.OUT_SLOTS:>3}  equal={equal} maxabs={maxabs:g}  "
              f"ref {ms_ref:.3f} ms   kernel {ms_got:.3f} ms   {ms_ref / ms_got:.2f}x  "
              f"({180.6 * H / 4 * (N / 3360) * (L / 3359) / ms_got:.0f} GB/s written)", flush=True)
        results[tag] = {
            "H": H, "L": L, "K": K, "N": N, "kind": kind, "slots": rfd3_bias.OUT_SLOTS,
            "bit_exact": equal, "maxabs": maxabs,
            "ms_ref_median": ms_ref, "ms_kernel_median": ms_got,
            "speedup": ms_ref / ms_got,
            "ms_ref_all": all_ref, "ms_kernel_all": all_got,
            "addr_write_mode": rfd3_bias.ADDR_WRITE_MODE,
        }
    for t in (pair_bias, idx_tiled, idx_rm, template):
        ttnn.deallocate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/p36/bias_kernel.json")
    ap.add_argument("--shapes", default="atom,dit")
    ap.add_argument("--kinds", default="random,banded,mixed")
    ap.add_argument("--slots", default="", help="comma-separated OUT_SLOTS sweep")
    args = ap.parse_args()

    slots = tuple(int(s) for s in args.slots.split(",")) if args.slots else (None,)
    kinds = tuple(args.kinds.split(","))
    rfd3_bias.set_enabled(True)
    device = ttnn.open_device(device_id=0)
    results: dict = {}
    try:
        want = args.shapes.split(",")
        if "atom" in want:
            for kind in kinds:
                one_shape(device, 4, 3359, 128, kind, results, slots)
        if "dit" in want:
            for kind in kinds:
                if kind != "mixed":
                    one_shape(device, 16, 250, 32, kind, results, slots)
    finally:
        ttnn.close_device(device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    bad = [k for k, v in results.items() if not v["bit_exact"]]
    print("NOT bit-exact: " + (", ".join(bad) if bad else "none"))


if __name__ == "__main__":
    main()
