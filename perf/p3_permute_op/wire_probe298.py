#!/usr/bin/env python3
"""The op at the fold's OWN shape, [1, 298, 298, 32], after the ragged-N fix.

The first wiring served zero calls in a whole fold because the kernel required N % 32 == 0 and the
production tensor is 298 wide. Everything here is at 298 unless it says otherwise, with 320 kept as
the second N so the stale-descriptor test still runs two different N through one cache.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn
from tt_bio import reblock_permute as RP


def timeit(device, fn, reps=15, warmup=3):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def host_only_us(x, device, mc, reps=200):
    N = int(x.shape[1])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 32, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
        entry = RP._prepare(x, out, device)
        pd = entry["pd"]
        pd.kernels[0].common_runtime_args = [x.buffer_address()]
        pd.kernels[1].common_runtime_args = [out.buffer_address()]
        ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/p3_permute_op/wire_probe298.json")
    a = ap.parse_args()
    RP.set_enabled(True)
    device = ttnn.open_device(device_id=0)
    R = {"rows": [], "checks": {}}

    refs, xs, gold = {}, {}, {}
    for N in (298, 320):
        ref = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
        refs[N] = ref
        gold[N] = ref.permute(0, 3, 1, 2).contiguous()
        xs[N] = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                                memory_config=ttnn.L1_MEMORY_CONFIG)

    seq = []
    for _ in range(3):
        for N in (298, 320, 320, 298):
            r = RP.reblock_permute(xs[N], ttnn.L1_MEMORY_CONFIG, device)
            seq.append({"N": N, "torch_equal": bool(torch.equal(ttnn.to_torch(r), gold[N]))})
            ttnn.deallocate(r)
    R["checks"]["alternating_N_298_320"] = seq
    R["checks"]["all_equal"] = all(s["torch_equal"] for s in seq)
    R["checks"]["cache_entries"] = len(RP._CACHE)

    # The output's tile padding must be ZERO: it sits on the contracted axis of the matmul that
    # consumes this tensor, so a copy of row 0 there would change the product. Read the padded
    # buffer back through a reshape-free path: compare against ttnn.permute's own padded output.
    N = 298
    x = xs[N]
    ours = RP.reblock_permute(x, ttnn.L1_MEMORY_CONFIG, device)
    theirs = ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    pad_ok = bool(torch.equal(ttnn.to_torch(ours), ttnn.to_torch(theirs)))
    R["checks"]["equal_to_ttnn_permute_298"] = pad_ok
    ttnn.deallocate(ours); ttnn.deallocate(theirs)

    for N in (298, 320):
        for where in ("l1", "dram"):
            mc = ttnn.L1_MEMORY_CONFIG if where == "l1" else ttnn.DRAM_MEMORY_CONFIG
            xx = ttnn.from_torch(refs[N], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                 device=device, memory_config=mc)
            r = RP.reblock_permute(xx, mc, device)
            eq = torch.equal(ttnn.to_torch(r), gold[N])
            ttnn.deallocate(r)

            def wired(xx=xx, mc=mc):
                o = RP.reblock_permute(xx, mc, device); ttnn.deallocate(o)

            def base(xx=xx, mc=mc):
                o = ttnn.permute(xx, (0, 3, 1, 2), memory_config=mc); ttnn.deallocate(o)

            uw, ub = timeit(device, wired), timeit(device, base)
            row = {"N": N, "buf": where, "wired_wall_us": round(uw, 2),
                   "ttnn_permute_us": round(ub, 2), "ratio_wall": round(ub / uw, 3),
                   "host_per_call_cached_us": round(host_only_us(xx, device, mc), 2),
                   "torch_equal": bool(eq)}
            R["rows"].append(row); print(row, flush=True)
            ttnn.deallocate(xx)

    ttnn.close_device(device)
    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    print(json.dumps(R["checks"], indent=2))


if __name__ == "__main__":
    main()
