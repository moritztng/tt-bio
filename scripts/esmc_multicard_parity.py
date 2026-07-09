"""On-hardware parity gate for ESMC multi-card (data-parallel) embedding fanout.

The fanout splits a sequence set across N physical cards (one pinned subprocess
per card), embeds each shard with the single-card ``embed_sequences``, and
reassembles the results in original input order. This harness proves that layer
is lossless — the real risk is host-side (off-by-one shard boundaries, reorder
bugs, transport), not the device kernel, which is unchanged.

Two levels of check:

  1. HOST LOGIC (no device): ``_shard_by_length`` covers every id exactly once
     with balanced shard sizes, and ``_reassemble`` restores input order from
     out-of-order shard results. Exhaustive over many (N, shards) combos.

  2. ON-DEVICE bit-exact (card pinned via TT_VISIBLE_DEVICES): with ``batch_size=1``
     each sequence is embedded alone in its own length bucket, so a sharded run is
     bit-exact vs the single-shot single-card run *by construction* (no batchmate
     regrouping — cf. the esmc-embed-batching row-independence method). We compute
     the single-shot reference and every shard in short-lived pinned subprocesses
     (so the parent never holds the card and can drive shards sequentially on ONE
     leased card), then assert Δ == 0 per-residue / pooled / logits and identical
     ids+order. Concurrent multi-card wall-clock scaling needs 2+ free cards and is
     out of scope here (documented as the follow-up leg).

Usage:
    TT_VISIBLE_DEVICES=3 python3 scripts/esmc_multicard_parity.py \
        --model esmc-300m --n 24 --shards 4
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from tt_bio import esmc as tt_esmc


def make_seqs(n: int, seed: int = 7) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    aa = np.array(list("LAGVSERTIDPKQNFYMHWC"))
    seqs = {}
    for i in range(n):
        L = int(rng.integers(40, 121))
        seqs[f"seq{i}_len{L}"] = "".join(rng.choice(aa, size=L))
    return seqs


def check_host_logic(seed: int = 0) -> None:
    """Exhaustively verify shard coverage/balance + reassembly order (no device)."""
    rng = np.random.default_rng(seed)
    for n in [1, 2, 3, 7, 16, 24, 50]:
        items = [(f"id{i}", "A" * int(rng.integers(1, 200))) for i in range(n)]
        for shards_n in [1, 2, 3, 4, 8, n, n + 3]:
            shards = tt_esmc._shard_by_length(items, shards_n)
            flat = [sid for sh in shards for sid, _ in sh]
            assert sorted(flat) == sorted(sid for sid, _ in items), \
                f"coverage broken n={n} shards={shards_n}"
            assert len(flat) == len(set(flat)), f"duplicate id n={n} shards={shards_n}"
            sizes = [len(sh) for sh in shards if sh]
            if sizes:
                assert max(sizes) - min(sizes) <= 1, \
                    f"unbalanced shards n={n} shards={shards_n}: {sizes}"
            # reassembly: shuffle per-shard fake results, must restore input order
            fake = []
            for sh in shards:
                rows = [_FakeEmb(sid) for sid, _ in sh]
                rng.shuffle(rows)
                fake.append(rows)
            rng.shuffle(fake)
            out = tt_esmc._reassemble(items, fake)
            assert [e.id for e in out] == [sid for sid, _ in items], \
                f"order broken n={n} shards={shards_n}"
    print("[host] shard coverage / balance / reassembly-order: OK")


class _FakeEmb:
    def __init__(self, sid):
        self.id = sid


def run_shard_subprocess(device, seqs, model, workdir, idx, *, return_logits, batch_size):
    """Run one shard through the production _spawn_shard/_await_shard, serially."""
    h = tt_esmc._spawn_shard(idx, device, list(seqs.items()), workdir, model=model,
                             fast=False, return_logits=return_logits, pool="mean",
                             batch_size=batch_size)
    return tt_esmc._await_shard(*h)


def max_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=list(tt_esmc.CONFIGS))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--logits", action="store_true")
    args = ap.parse_args()

    check_host_logic()

    device = int(os.environ.get("TT_VISIBLE_DEVICES", "0"))
    seqs = make_seqs(args.n)
    items = list(seqs.items())
    workdir = tempfile.mkdtemp(prefix="tt-bio-multicard-parity-")
    print(f"[device {device}] model={args.model} n={args.n} shards={args.shards} "
          f"logits={args.logits} bs=1 (per-sequence bucketing -> bit-exact by construction)")

    # Reference: the single-shot single-card path, as a pinned subprocess (so the
    # parent never opens the card and the shard subprocesses can reuse it serially).
    ref = run_shard_subprocess(device, seqs, args.model, workdir, 999,
                               return_logits=args.logits, batch_size=1)
    ref = {e.id: e for e in ref}
    print(f"[ref] single-shot single-card: {len(ref)} embeddings")

    # Fanout: split into `shards`, run each pinned subprocess sequentially on the
    # one leased card, reassemble in input order.
    shards = tt_esmc._shard_by_length(items, args.shards)
    shard_results = []
    for idx, sh in enumerate(shards):
        if not sh:
            continue
        r = run_shard_subprocess(device, dict(sh), args.model, workdir, idx,
                                 return_logits=args.logits, batch_size=1)
        shard_results.append(r)
        print(f"[shard {idx}] {len(sh)} seq -> {len(r)} embeddings "
              f"(lens {sorted(len(s) for _, s in sh)})")
    fan = tt_esmc._reassemble(items, shard_results)

    # 1. ids + order identical to input
    assert [e.id for e in fan] == [sid for sid, _ in items], "fanout order != input order"
    # 2. bit-exact vs single-shot reference
    worst_pr = worst_pool = worst_lg = 0.0
    for e in fan:
        r = ref[e.id]
        worst_pr = max(worst_pr, max_abs(e.per_residue, r.per_residue))
        worst_pool = max(worst_pool, max_abs(e.pooled, r.pooled))
        assert e.per_residue.shape == r.per_residue.shape
        if args.logits:
            worst_lg = max(worst_lg, max_abs(e.logits, r.logits))

    print(f"[parity] order: OK  |  Δmax per_residue={worst_pr:g}  pooled={worst_pool:g}"
          + (f"  logits={worst_lg:g}" if args.logits else ""))
    bit_exact = worst_pr == 0.0 and worst_pool == 0.0 and (not args.logits or worst_lg == 0.0)
    if not bit_exact:
        print("FAIL: fanout is NOT bit-exact vs single-shot single-card")
        return 1
    print("PASS: multi-card fanout is BIT-EXACT vs single-shot single-card "
          "(split + subprocess transport + gather + reorder are lossless)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
