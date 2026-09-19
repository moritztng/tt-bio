#!/usr/bin/env python3
"""`reblock_permute_gated` at the shape a 512 aa fold actually calls it: parity, then A/B.

The shape comes from `firing_512_qb1c0.json`, not from an op catalogue: xw [1,512,512,512],
slice_c 128 (Ct = 4), Ctw = 16, p/g slice offsets 8/12 tiles and 0/4 tiles, 1120 calls a fold.

Parity first. The shipped kernel (CT_STREAM = 1) is pinned bit-exact against the two-op ttnn
sequence by perf/trimul_f2/e6_parity.py, so both are checked here: every arm against the ttnn
reference and every arm against CT_STREAM = 1.

Timing is a batch of `n` calls between two synchronize_device calls, so the ~0.05 ms host bracket
is paid once per batch and not once per call. Arms are interleaved round by round, never
all-A-then-all-B, and one arm is a duplicate of CT_STREAM = 1 so the table carries its own A/A
floor. The AICLK is read on the measuring thread right after each batch's sync.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

N = 512
CW = 512
SLICE_C = 128
P_SLICE, G_SLICE = 8 * 32, 12 * 32


def clocks():
    out = {}
    for n in range(4):
        try:
            out[str(n)] = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text())
        except Exception:
            out[str(n)] = -1
    return out


def main():
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio import reblock_permute as rbp

    dev = get_device()
    grid = dev.compute_with_storage_grid_size()
    mc = ttnn.DRAM_MEMORY_CONFIG

    torch.manual_seed(0)
    xt = (torch.randn(1, N, N, CW) * 2.0).to(torch.bfloat16)
    xw = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=mc)

    # the two-op ttnn sequence this kernel replaces, as the float reference
    ch = ttnn.chunk(xw, 4, dim=-1)
    p_i, g_i = P_SLICE // SLICE_C, G_SLICE // SLICE_C
    gated = ttnn.multiply(ch[p_i], ttnn.sigmoid(ch[g_i], memory_config=mc), memory_config=mc)
    ref = ttnn.permute(gated, (0, 3, 1, 2), memory_config=mc)
    ttnn.deallocate(gated)
    ref_t = ttnn.to_torch(ref)
    ttnn.deallocate(ref)
    for c in ch:
        ttnn.deallocate(c)

    # `S:R:bufs` per arm. Repeat an arm to give the table its own A/A floor.
    arms = [tuple(int(v) for v in spec.split(":")) for spec in sys.argv[2].split(",")]
    names = []
    for i, (S, R, b) in enumerate(arms):
        names.append(f"{i}_S{S}_R{R}_b{b}")

    out_buf = {}

    def run(S, R, bufs, n):
        rbp.CT_STREAM, rbp.CT_BUFS, rbp.ROW_BATCH = S, bufs, R
        key = (S, R, bufs)
        if key not in out_buf:
            out_buf[key] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, SLICE_C, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)
        o = out_buf[key]
        for _ in range(n):
            rbp.reblock_permute_gated(xw, P_SLICE, G_SLICE, SLICE_C, memory_config=mc,
                                      device=dev, out=o, row_off=0)
        return o

    # --- parity -------------------------------------------------------------------------------
    parity = {}
    base_t = None
    for nm, (S, R, b) in zip(names, arms):
        o = run(S, R, b, 1)
        ttnn.synchronize_device(dev)
        t = ttnn.to_torch(o)
        if base_t is None:
            base_t = t
        parity[nm] = {
            "equal_to_ttnn_reference": bool(torch.equal(t, ref_t)),
            "equal_to_first_arm": bool(torch.equal(t, base_t)),
            "max_abs_diff_vs_ttnn": float((t.float() - ref_t.float()).abs().max()),
            "ct_stream_used": rbp._ct_stream(SLICE_C // 32),
        }

    # --- timing -------------------------------------------------------------------------------
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    for (S, R, b) in arms:            # warm: compile + program cache, discarded
        run(S, R, b, 3)
    ttnn.synchronize_device(dev)

    samples = {nm: [] for nm in names}
    clk = []
    for _ in range(rounds):
        for nm, (S, R, b) in zip(names, arms):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            run(S, R, b, n)
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            clk.append(clocks())
            samples[nm].append(dt / n * 1e3)   # ms per call

    med = {k: round(statistics.median(v), 5) for k, v in samples.items()}
    base = med[names[0]]
    out = {
        "shape": {"xw": [1, N, N, CW], "slice_c": SLICE_C, "Ct": SLICE_C // 32,
                  "Ctw": CW // 32, "p_off_tiles": P_SLICE // 32, "g_off_tiles": G_SLICE // 32},
        "host": os.uname().nodename, "grid": [grid.x, grid.y],
        "batch": n, "rounds": rounds,
        "parity": parity,
        "ms_per_call_median": med,
        "ms_per_call_all": {k: [round(x, 5) for x in v] for k, v in samples.items()},
        "speedup_vs_first_arm": {k: round(base / v, 5) for k, v in med.items()},
        "aiclk_during": {str(i): sorted(set(c[str(i)] for c in clk)) for i in range(4)},
        "loadavg": [round(v, 2) for v in os.getloadavg()],
    }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
