"""Price the three per-step costs hidden inside `_sparse_qk_inputs`.

The stage profiler attributes 68 ms of a 427 ms step (250 residues, D=1) to this one
helper, which p9 never separated from the 7.5 ms host pair gather it wraps. This
prices the other two pieces -- the -1e4 dense-bias `ttnn.full` and the attention-index
upload -- against the alternatives, and checks the invariant a cached dense bias
depends on: that `ttnn.scatter` does not write through to its input.

Run with the standard worker env (TT_VISIBLE_DEVICES=0, PYTHONPATH=<worktree>).
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ttnn  # noqa: E402

from tt_bio.rfd3 import _sparse_qk_host, _tt  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

L, H, K, B = 3359, 4, 128, 1
REPS = 6


def timed(fn, dev, reps=REPS):
    fn()
    ttnn.synchronize_device(dev)
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        samples.append((time.perf_counter() - start) * 1e3)
        del out
    return statistics.median(samples)


def main() -> None:
    dev = get_device()
    torch.manual_seed(0)
    indices = torch.stack(
        [torch.randperm(L)[:K] for _ in range(L)]
    ).unsqueeze(0)  # [1,L,K]
    p_host = torch.randn(1, L, L, 16)

    print(f"L={L} H={H} K={K} B={B}  dense bias = {B*H*L*L*2/1e6:.0f} MB bf16")

    # --- 1. the -1e4 dense bias -------------------------------------------------
    full_ms = timed(
        lambda: ttnn.full((B, H, L, L), -1e4, dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=dev), dev)
    print(f"ttnn.full((B,H,L,L), -1e4)                 {full_ms:8.3f} ms   <- per step today")

    # A cached bias is only sound if scatter is out-of-place. Check it directly:
    # scatter, then read the SOURCE back and confirm it is still -1e4 everywhere.
    dense_bias = ttnn.full((B, H, L, L), -1e4, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev)
    attn_idx = _tt_idx_tile(indices.unsqueeze(1).expand(B, H, L, K).to(torch.int32), dev)
    pair_bias = _tt(torch.randn(B, H, L, K), dev, ttnn.bfloat16)
    scattered = ttnn.scatter(dense_bias, 3, attn_idx, pair_bias)
    src_after = ttnn.to_torch(dense_bias).float()
    out_after = ttnn.to_torch(scattered).float()
    src_pristine = bool((src_after == -1e4).all())
    out_changed = bool((out_after != -1e4).any())
    print(f"scatter leaves its input pristine          {src_pristine}   "
          f"(output actually changed: {out_changed})")

    # --- 2. the attention-index upload -----------------------------------------
    attn_host = indices.unsqueeze(1).expand(B, H, L, K).to(torch.int32).contiguous()
    host_tilize_ms = timed(
        lambda: ttnn.from_torch(attn_host, layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.uint32), dev)

    def dev_tilize():
        rm = ttnn.from_torch(attn_host, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                             dtype=ttnn.uint32)
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    dev_tilize_ms = timed(dev_tilize, dev)
    a = ttnn.to_torch(ttnn.from_torch(attn_host, layout=ttnn.TILE_LAYOUT, device=dev,
                                      dtype=ttnn.uint32))
    b = ttnn.to_torch(dev_tilize())
    print(f"attn_idx upload, host tilize (shipped)     {host_tilize_ms:8.3f} ms")
    print(f"attn_idx upload, device tilize             {dev_tilize_ms:8.3f} ms   "
          f"equal={torch.equal(a, b)}")

    # --- 3. the host pair gather ------------------------------------------------
    start = time.perf_counter()
    for _ in range(REPS):
        _sparse_qk_host(p_host, indices)
    gather_ms = (time.perf_counter() - start) * 1e3 / REPS
    print(f"_sparse_qk_host (pure host)                {gather_ms:8.3f} ms   "
          f"x3 calls/step today")

    total = full_ms + host_tilize_ms + gather_ms
    saved = full_ms + (host_tilize_ms - dev_tilize_ms) + gather_ms
    print(f"\naccounted: {total:.1f} ms of the 68 ms the stage profiler attributes")
    print(f"removable bit-exactly (cache bias + device tilize + drop 1 of 3 gathers): "
          f"{saved:.1f} ms/step")


def _tt_idx_tile(x, dev):
    return ttnn.from_torch(x.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.uint32)


if __name__ == "__main__":
    main()
