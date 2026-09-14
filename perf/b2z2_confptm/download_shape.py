#!/usr/bin/env python3
"""How three numbers per token pair should cross the bus.

``ConfidenceHeadsDevice`` ends with a ``[1, n_pad, n_pad, 32]`` bf16 device tensor whose three
used channels are pae, the TM expectation and pde. Getting those to the host has a choice, and
the stage wall inside a fold cannot resolve it: across three warm folds the download stage read
18.31, 18.32 and 20.53 ms, which is wider than the difference between the candidates. So time it
standalone and interleaved, n reps, on a tensor of exactly the shape the head produces.

  tiled     ``to_torch`` on the TILE tensor, slice on the host.            16.8 MB on the bus.
  narrow    ``to_layout(ROW_MAJOR)`` + ``slice`` to 4 channels + ``to_torch``.  2.1 MB.
  untiled   ``to_layout(ROW_MAJOR)`` + ``to_torch``, slice on the host.    16.8 MB, no device slice.

Every variant must return the same three channels; that is asserted, not assumed. No variant
copies its input first -- an earlier revision cloned per rep to be safe, and the clone was a
16.8 MB device copy that landed on all three and inflated every absolute number away from what
the fold's own stage wall reads. The source tensor is asserted unchanged at the end instead.

Sub-steps are timed separately because the head's stage wall brackets ``to_layout`` and
``slice + to_torch`` on different sides of one mark, so only a split table can be laid against it.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CH_KEEP = 4
CH_PAE, CH_TM, CH_PDE = 0, 1, 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=512, help="tokens (the unpadded seq_len)")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio.tenstorrent import get_device

    torch.set_grad_enabled(False)
    device = get_device()
    padded = (a.n + 31) // 32 * 32

    # the head's own output shape, with distinguishable content in every channel
    src = torch.arange(padded * padded * 32, dtype=torch.float32).reshape(1, padded, padded, 32)
    src = (src % 997) / 997.0
    tile = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    src_dev = torch.Tensor(ttnn.to_torch(tile)).to(torch.float32)   # bf16-rounded, the real datum

    def keep(t: "torch.Tensor") -> "torch.Tensor":
        return t[:, :a.n, :a.n, :CH_KEEP].contiguous().to(torch.float32)

    def step(acc, name):
        """Time one sub-step, charging device time to it via a sync on both sides."""
        class _T:
            def __enter__(self):
                ttnn.synchronize_device(device)
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, *exc):
                ttnn.synchronize_device(device)
                acc.setdefault(name, []).append(1e3 * (time.perf_counter() - self.t0))
                return False
        return _T()

    def tiled(acc):
        with step(acc, "to_torch"):
            t = torch.Tensor(ttnn.to_torch(tile))
        return keep(t)

    def narrow(acc):
        with step(acc, "to_layout"):
            rm = ttnn.to_layout(tile, ttnn.ROW_MAJOR_LAYOUT)
        with step(acc, "slice"):
            sl = ttnn.slice(rm, [0, 0, 0, 0], [1, a.n, a.n, CH_KEEP])
        with step(acc, "to_torch"):
            out = torch.Tensor(ttnn.to_torch(sl)).to(torch.float32)
        ttnn.deallocate(sl)
        ttnn.deallocate(rm)
        return out

    def untiled(acc):
        with step(acc, "to_layout"):
            rm = ttnn.to_layout(tile, ttnn.ROW_MAJOR_LAYOUT)
        with step(acc, "to_torch"):
            t = torch.Tensor(ttnn.to_torch(rm))
        ttnn.deallocate(rm)
        return keep(t)

    variants = {"tiled": tiled, "narrow": narrow, "untiled": untiled}
    steps: dict = {k: {} for k in variants}

    # correctness first: the bus shape is a choice, the numbers are not
    ref = tiled({})
    for name, fn in variants.items():
        got = fn({})
        assert got.shape == ref.shape, f"{name}: {got.shape} != {ref.shape}"
        d = (got - ref).abs().max().item()
        assert d == 0.0, f"{name}: max abs {d} against tiled"

    bus = {"tiled": padded * padded * 32 * 2,
           "narrow": a.n * a.n * CH_KEEP * 2,
           "untiled": padded * padded * 32 * 2}

    for fn in variants.values():           # warm the first-call compile out of the timings
        for _ in range(2):
            fn({})

    ms: dict = {k: [] for k in variants}
    for _ in range(a.reps):                # interleaved, so host drift lands on all three
        for name, fn in variants.items():
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            fn(steps[name])
            ttnn.synchronize_device(device)
            ms[name].append(1e3 * (time.perf_counter() - t0))

    # no variant may mutate the tensor it reads -- that is what lets them share one source
    after = torch.Tensor(ttnn.to_torch(tile)).to(torch.float32)
    drift = (after - src_dev).abs().max().item()
    assert drift == 0.0, f"source tensor moved by {drift} across the run"

    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "n": a.n, "padded": padded, "reps": a.reps,
                   "loadavg": list(os.getloadavg())},
           "variants": {}}
    for name in variants:
        v = ms[name]
        out["variants"][name] = {"median_ms": round(st.median(v), 3),
                                 "min_ms": round(min(v), 3),
                                 "max_ms": round(max(v), 3),
                                 "bus_bytes": bus[name],
                                 "bus_mb": round(bus[name] / 1e6, 3),
                                 "steps_median_ms": {k: round(st.median(w), 3)
                                                     for k, w in steps[name].items()}}
        r = out["variants"][name]
        parts = "  ".join(f"{k} {x:.3f}" for k, x in r["steps_median_ms"].items())
        print(f"  {name:8s} median {r['median_ms']:7.3f} ms  min {r['min_ms']:7.3f}  "
              f"max {r['max_ms']:7.3f}  bus {r['bus_mb']:6.3f} MB   [{parts}]", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
