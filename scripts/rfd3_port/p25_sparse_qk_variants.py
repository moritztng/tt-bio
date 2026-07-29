"""p25: measure the two host round-trips inside `_sparse_qk_inputs` before porting either.

p21's dedup made this one build per step instead of three, and p24's lesson is that three
passes assumed `attn_indices` wanted an on-device port when the real lever was allocation.
So price the pieces first.

What the shipped path does per step, once the mask template is cached:
  1. `_sparse_qk_host` -- a torch advanced-index gather p_host[b, i, indices[b,i,k]] into a
     (B, L, K, 16) fp32 tensor, on host;
  2. upload that as bf16 (13.8 MB at 3359 atoms, D=1);
  3. upload the scatter index, expanded over the 4 heads on host first, as uint32
     (6.9 MB -- 4x the (B, L, K) it is built from).

P_LL is produced once by the TokenInitializer and never changes across steps or recycles, so
it can live on device as a (L*L, 16) bf16 gather table and step 1+2 become one
`ttnn.embedding`. The candidates, measured separately so a win can be attributed:

  G  device gather -- resident P_LL table + a flat (B*L*K) uint32 index upload, replacing the
     host gather and the p_sparse upload. Bit-exact by construction: a gather is a copy, and
     bf16(P_LL)[idx] == bf16(P_LL[idx]) elementwise. Checked, not asserted.
  H  head expand on device -- upload the index once as (B, 1, L, K) and replicate over heads
     with a concat, so only a quarter of those bytes cross the boundary.

Both are measured against the shipped path at the same sizes, and every candidate's output is
compared bit-for-bit with what the shipped path produces.

Usage:
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:... PYTHONPATH=$WT \
    python3 scripts/rfd3_port/p25_sparse_qk_variants.py --batches 1 8 --atoms 3359 2702
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ttnn  # noqa: E402

from tt_bio.rfd3 import _sparse_qk_host, _tt  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--atoms", type=int, nargs="+", default=[3359, 2702])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
ap.add_argument("--keys", type=int, default=128)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--channels", type=int, default=16)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

DEV = get_device()


def timed(fn, reps=None):
    """Median wall time in ms, device synchronized, first call discarded."""
    reps = reps or args.reps
    out = fn()
    ttnn.synchronize_device(DEV)
    del out
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(DEV)
        samples.append((time.perf_counter() - start) * 1e3)
        del out
    return statistics.median(samples)


def run(L: int, B: int) -> None:
    H, K, C = args.heads, args.keys, args.channels
    torch.manual_seed(0)
    # P_LL is batch-1 and step-invariant (TokenInitializer output, rfd3.py:823).
    p_host = torch.randn(1, L, L, C)
    indices = torch.stack([torch.randperm(L)[:K] for _ in range(L)])
    indices = indices.unsqueeze(0).expand(B, -1, -1).contiguous()

    table_mb = L * L * C * 2 / 1e6
    print(f"\n=== L={L} B={B} K={K} H={H} C={C} | resident P_LL table {table_mb:.0f} MB bf16 "
          f"| p_sparse {B*L*K*C*2/1e6:.1f} MB bf16 | attn_idx {B*H*L*K*4/1e6:.1f} MB uint32")

    # ---- shipped path, piece by piece --------------------------------------
    host_gather_ms = timed(lambda: _sparse_qk_host(p_host, indices, H), reps=args.reps)
    p_sparse, attn_idx, _ = _sparse_qk_host(p_host, indices, H)
    upload_p_ms = timed(lambda: _tt(p_sparse, DEV, ttnn.bfloat16))
    upload_idx_ms = timed(lambda: _tt(attn_idx, DEV, ttnn.uint32))
    shipped_ms = host_gather_ms + upload_p_ms + upload_idx_ms
    print(f"  shipped  host gather {host_gather_ms:7.2f}  upload p_sparse {upload_p_ms:7.2f}  "
          f"upload attn_idx {upload_idx_ms:7.2f}  total {shipped_ms:7.2f} ms")
    reference = ttnn.to_torch(_tt(p_sparse, DEV, ttnn.bfloat16)).float()

    # ---- G: resident table + device gather ---------------------------------
    # One-time: P_LL as a (L*L, C) bf16 row-major gather table.
    t0 = time.perf_counter()
    table = ttnn.from_torch(p_host.reshape(L * L, C).contiguous(),
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(DEV)
    table_build_ms = (time.perf_counter() - t0) * 1e3

    row_offset = (torch.arange(L, dtype=torch.int64) * L).reshape(1, L, 1)

    def flat_index():
        return (indices.long() + row_offset).reshape(1, B * L * K).to(torch.int32)

    flat_ms = timed(lambda: flat_index(), reps=args.reps)

    def gather_device():
        idx = ttnn.from_torch(flat_index(), layout=ttnn.ROW_MAJOR_LAYOUT, device=DEV,
                              dtype=ttnn.uint32)
        got = ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        got = ttnn.reshape(got, (B, L, K, C))
        return ttnn.to_layout(got, ttnn.TILE_LAYOUT)

    gather_ms = timed(gather_device)
    got = ttnn.to_torch(gather_device()).float()
    exact = bool(torch.equal(got, reference))
    print(f"  G        flat index  {flat_ms:7.2f}  device gather   {gather_ms:7.2f}  "
          f"(table build {table_build_ms:.0f} ms one-time)     "
          f"total {gather_ms:7.2f} ms  bit-exact={exact} "
          f"maxabs={(got - reference).abs().max().item():.3e}")
    replaced = host_gather_ms + upload_p_ms
    print(f"           replaces {replaced:.2f} ms -> {gather_ms:.2f} ms  "
          f"delta {replaced - gather_ms:+.2f} ms/step")

    # ---- H: upload the index once, replicate over heads on device ----------
    idx_1h = indices.unsqueeze(1).to(torch.int32).contiguous()          # (B,1,L,K)

    def expand_device():
        up = _tt(idx_1h, DEV, ttnn.uint32)
        return ttnn.concat([up] * H, dim=1) if H > 1 else up

    expand_ms = timed(expand_device)
    idx_ref = ttnn.to_torch(_tt(attn_idx, DEV, ttnn.uint32))
    idx_got = ttnn.to_torch(expand_device())
    idx_exact = bool(torch.equal(idx_got, idx_ref))
    print(f"  H        upload (B,1,L,K) + device concat x{H}: {expand_ms:7.2f} ms  "
          f"replaces {upload_idx_ms:.2f} ms  delta {upload_idx_ms - expand_ms:+.2f} ms/step  "
          f"bit-exact={idx_exact}")

    best = gather_ms + min(expand_ms, upload_idx_ms) + flat_ms
    print(f"  => shipped {shipped_ms:.2f} ms/step vs best candidate {best:.2f} ms/step "
          f"({shipped_ms - best:+.2f} ms/step, {100 * (shipped_ms - best) / shipped_ms:+.1f}%)")


for L in args.atoms:
    for B in args.batches:
        run(L, B)
