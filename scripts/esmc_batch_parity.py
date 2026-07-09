"""On-hardware parity + throughput gate for ESMC embedding batching.

Verifies that the batched, length-bucketed embedding path (``embed_sequences``
with ``batch_size > 1``) reproduces the original one-sequence-at-a-time,
unbucketed forward bit-for-bit-within-noise, and measures the throughput win on
a realistic FASTA of many short sequences.

The reference here is the *pre-batching* device path itself: raw ``model(tokens)``
with no padding and no mask (exactly what the shipped CLI did before), so this
isolates the batching/masking change from any esm-reference discrepancy already
covered by esmc_embed_parity.py.

Usage:
    TT_VISIBLE_DEVICES=0 python3 scripts/esmc_batch_parity.py --model esmc-300m \
        --n 32 --batch-size 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from tt_bio import esmc as tt_esmc


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def make_seqs(n: int, seed: int = 42) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    aa = np.array(list("LAGVSERTIDPKQNFYMHWC"))
    seqs = {}
    for i in range(n):
        L = int(rng.integers(40, 121))
        seqs[f"seq{i}_len{L}"] = "".join(rng.choice(aa, size=L))
    return seqs


def reference_unbatched(model, seqs: dict[str, str], pool: str):
    """Original path: one unpadded, unmasked forward per sequence."""
    out = {}
    t0 = time.time()
    for sid, seq in seqs.items():
        model.reset_static_cache()
        _, em = model(tt_esmc.tokenize(seq))           # [1, L+2, d]
        emb = em[0].numpy().astype(np.float32)
        per_residue, cls = emb[1:-1], emb[0]
        pooled = cls if pool == "cls" else tt_esmc._POOLERS[pool](per_residue)
        out[sid] = (per_residue, pooled.astype(np.float32))
    return out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=list(tt_esmc.CONFIGS))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pool", default="mean", choices=["mean", "max", "cls"])
    # 0.99 matches the shipped ESMC embedding accuracy gate (esmc_embed_parity.py).
    # Batching adds only (B, Lb)-shape-driven fp32-accumulation reordering on top
    # of the unbatched device path — the noise class the 6B bucketed path already
    # accepts — so per-residue PCC stays well above it (300M ~0.9993, 600M ~0.997).
    ap.add_argument("--pcc-threshold", type=float, default=0.99)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    seqs = make_seqs(args.n)
    print(f"[{args.model}] {len(seqs)} seqs, lengths "
          f"{min(len(s) for s in seqs.values())}–{max(len(s) for s in seqs.values())}, "
          f"batch_size={args.batch_size}", flush=True)

    m = tt_esmc.load_esmc(args.model, fast=args.fast)

    # --- reference: original one-at-a-time, unbucketed forward ---
    ref, t_ref = reference_unbatched(m, seqs, args.pool)
    print(f"reference (per-seq, unbucketed): {t_ref:.1f}s  "
          f"{len(seqs) / t_ref:.3f} seq/s", flush=True)

    # --- batched path, cold (first time these bucketed shapes are compiled) ---
    t0 = time.time()
    batched = tt_esmc.embed_sequences(m, seqs, pool=args.pool,
                                      batch_size=args.batch_size)
    t_batch = time.time() - t0
    print(f"batched (bs={args.batch_size}) cold:  {t_batch:5.1f}s  "
          f"{len(seqs) / t_batch:.3f} seq/s   speedup {t_ref / t_batch:.2f}x",
          flush=True)

    # --- batched path, warm (all bucket shapes now cached: steady-state) ---
    t0 = time.time()
    tt_esmc.embed_sequences(m, seqs, pool=args.pool, batch_size=args.batch_size)
    t_warm = time.time() - t0
    print(f"batched (bs={args.batch_size}) warm:  {t_warm:5.1f}s  "
          f"{len(seqs) / t_warm:.3f} seq/s   speedup {t_ref / t_warm:.2f}x",
          flush=True)

    # --- parity: per-residue + pooled PCC and max abs diff, per sequence ---
    worst_pr_pcc, worst_pool_pcc, worst_abs = 1.0, 1.0, 0.0
    for emb in batched:
        rp, rpool = ref[emb.id]
        pr_pcc = pcc(emb.per_residue, rp)
        pool_pcc = pcc(emb.pooled, rpool)
        abs_d = float(np.abs(emb.per_residue - rp).max())
        worst_pr_pcc = min(worst_pr_pcc, pr_pcc)
        worst_pool_pcc = min(worst_pool_pcc, pool_pcc)
        worst_abs = max(worst_abs, abs_d)

    print(f"\nworst per-residue PCC: {worst_pr_pcc:.6f}")
    print(f"worst pooled     PCC: {worst_pool_pcc:.6f}")
    print(f"worst |Δ| per-residue: {worst_abs:.4e}")

    # --- row independence: a row must be bit-exact regardless of its batchmates ---
    # Same batch shape (B, Lb), different other-row content -> identical output for
    # the probe row, proving the padding mask fully isolates each sequence (any
    # residual vs the unbatched path is only (B,Lb)-shape reordering noise, above).
    probe = list(seqs.values())[0]
    mate_len = max(len(s) for s in seqs.values())
    rng = np.random.default_rng(123)
    aa = np.array(list("LAGVSERTIDPKQNFYMHWC"))
    mate = lambda: "".join(rng.choice(aa, size=mate_len))
    ids_a, la, am_a, kv_a = tt_esmc._batch_tokens([probe, mate()])
    ids_b, lb, am_b, kv_b = tt_esmc._batch_tokens([probe, mate()])
    _, ea = m(ids_a, am_a, kv_a)
    _, eb = m(ids_b, am_b, kv_b)
    row_delta = float(np.abs(ea[0, :la[0]].numpy() - eb[0, :lb[0]].numpy()).max())
    print(f"row-independence |Δ| (different batchmates): {row_delta:.3e}")

    ok = (worst_pr_pcc >= args.pcc_threshold and worst_pool_pcc >= args.pcc_threshold
          and row_delta == 0.0)
    print(f"\n{'PASS' if ok else 'FAIL'} (PCC threshold {args.pcc_threshold}, "
          f"row-independence must be exact)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
