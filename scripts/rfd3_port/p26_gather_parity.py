"""p26: is the device gather inside `_sparse_qk_inputs` bit-identical to the host one?

p25 checked its candidate in a standalone micro-benchmark. This checks the shipped
function: run it with a cache (device gather off the resident P_LL table) and compare
what it returns against the host gather it replaced, tensor for tensor, at the real
shapes and batch sizes. maxabs must be exactly 0.0, and the uint32 scatter index must
be `torch.equal`.

Also checks the two fallbacks stay on the host path (fp32 pair stream, no cache) and
that a second call in the same step still hits the dedup cache.

Usage:
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:... PYTHONPATH=$WT \
    python3 scripts/rfd3_port/p26_gather_parity.py --atoms 3359 2702 --batches 1 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ttnn  # noqa: E402

from tt_bio.rfd3 import _sparse_qk_host, _sparse_qk_inputs, _tt  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--atoms", type=int, nargs="+", default=[3359, 2702])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
ap.add_argument("--keys", type=int, default=128)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--channels", type=int, default=16)
args = ap.parse_args()

DEV = get_device()
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(name)


for L in args.atoms:
    for B in args.batches:
        H, K, C = args.heads, args.keys, args.channels
        torch.manual_seed(0)
        p_host = torch.randn(1, L, L, C)
        idx = torch.stack([torch.randperm(L)[:K] for _ in range(L)])
        indices = idx.unsqueeze(0).expand(B, -1, -1).contiguous()
        print(f"\n=== L={L} B={B} K={K} H={H} C={C}")

        # host reference: exactly what the shipped path built before this pass
        p_sparse_ref, attn_idx_ref, n_keys_ref = _sparse_qk_host(p_host, indices, H)
        p_ref = ttnn.to_torch(_tt(p_sparse_ref, DEV, ttnn.bfloat16)).float()
        i_ref = ttnn.to_torch(_tt(attn_idx_ref, DEV, ttnn.uint32))

        # device path (a cache is what makes the resident table worth holding)
        cache = {}
        p_dev, n_keys, i_dev, dense_bias = _sparse_qk_inputs(
            p_host, indices, DEV, ttnn.bfloat16, n_heads=H, mask_cache=cache)
        check("resident table built", "tables" in cache and len(cache["tables"]) == 1)
        got_p = ttnn.to_torch(p_dev).float()
        got_i = ttnn.to_torch(i_dev)
        check("pair features shape", tuple(got_p.shape) == tuple(p_ref.shape),
              f"{tuple(got_p.shape)}")
        check("pair features bit-exact", bool(torch.equal(got_p, p_ref)),
              f"maxabs={(got_p - p_ref).abs().max().item():.3e}")
        check("scatter index bit-exact", bool(torch.equal(got_i, i_ref)))
        check("n_keys", n_keys == n_keys_ref, f"{n_keys}")
        check("dense_bias shape", tuple(dense_bias.shape)[:3] == (B, H, L))

        # the step dedup cache still returns the same objects for a repeat call
        again = _sparse_qk_inputs(p_host, indices, DEV, ttnn.bfloat16, n_heads=H,
                                  mask_cache=cache)
        check("step dedup hit", again[0] is p_dev and again[2] is i_dev)

        # a fresh view of the SAME data (what every step actually hands us) must reuse
        # the table rather than rebuild it
        view = p_host.squeeze(0).unsqueeze(0)
        _sparse_qk_inputs(view, indices, DEV, ttnn.bfloat16, n_heads=H, mask_cache=cache)
        check("table reused across a fresh view", len(cache["tables"]) == 1)

        # fp32 pair stream falls back to the host gather (ttnn.embedding is bf16-only)
        f32_cache = {}
        p32 = _sparse_qk_inputs(p_host, indices, DEV, ttnn.float32, n_heads=H,
                                mask_cache=f32_cache)[0]
        check("fp32 stays on host gather", "tables" not in f32_cache)
        check("fp32 values", bool(torch.equal(
            ttnn.to_torch(p32).float(),
            ttnn.to_torch(_tt(p_sparse_ref, DEV, ttnn.float32)).float())))

        # no cache at all (isolated component tests) also falls back
        p_nc = _sparse_qk_inputs(p_host, indices, DEV, ttnn.bfloat16, n_heads=H)[0]
        check("cacheless stays on host gather", bool(torch.equal(
            ttnn.to_torch(p_nc).float(), p_ref)))

print("\nRESULT:", "PARITY PASS" if not FAILURES else f"FAIL {FAILURES}")
sys.exit(1 if FAILURES else 0)
