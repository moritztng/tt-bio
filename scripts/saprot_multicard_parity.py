"""On-hardware parity gate for SaProt multi-card (data-parallel) embedding fanout.

Mirrors ``scripts/esmc_multicard_parity.py``. The fanout splits a sequence set
across N physical cards (one pinned subprocess per card), embeds each shard with
the single-card ``saprot.embed_sequences``, and reassembles in original input
order. This harness proves that layer is lossless — the real risk is host-side
(shard boundaries, reorder, transport), not the device kernel, which is unchanged.

Two levels of check:

  1. HOST LOGIC (no device): ``saprot._shard_by_length`` (aa-length key) covers every
     id exactly once with balanced shard sizes, and ``_reassemble`` restores input
     order from out-of-order shard results. Exhaustive over many (N, shards) combos.

  2. ON-DEVICE bit-exact (card pinned via TT_VISIBLE_DEVICES): with ``batch_size=1``
     each sequence is embedded alone in its own length bucket, so a sharded run is
     bit-exact vs the single-shot single-card run *by construction* (no batchmate
     regrouping — same bar ``tt-bio embed --devices`` holds, see
     ``scripts/esmc_multicard_parity.py``). We compute the single-shot reference and
     every shard in short-lived pinned subprocesses (so the parent never holds the
     card and can drive shards sequentially on ONE leased card), then assert
     Δ == 0 per-residue / pooled and identical ids+order. Concurrent multi-card
     wall-clock scaling needs 2+ free cards and is out of scope here.

Usage:
    TT_VISIBLE_DEVICES=0 python3 scripts/saprot_multicard_parity.py \\
        --model saprot-650m --n 12 --shards 4
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from tt_bio import saprot as tt_saprot


def make_seqs(n: int, seed: int = 7) -> dict:
    """Random {id: (aa, struc)} pairs; struc is all '#' (sequence-only SaProt)."""
    rng = np.random.default_rng(seed)
    aa = np.array(list("LAGVSERTIDPKQNFYMHWC"))
    seqs = {}
    for i in range(n):
        L = int(rng.integers(20, 121))
        aaseq = "".join(rng.choice(aa, size=L))
        seqs[f"seq{i}_len{L}"] = (aaseq, "#" * L)
    return seqs


def check_host_logic(seed: int = 0) -> None:
    """Exhaustively verify shard coverage/balance + reassembly order (no device)."""
    rng = np.random.default_rng(seed)
    aa_key = lambda it: len(it[1][0])
    for n in [1, 2, 3, 7, 16, 24, 50]:
        items = [(f"id{i}", ("A" * int(rng.integers(1, 200)), "#")) for i in range(n)]
        for shards_n in [1, 2, 3, 4, 8, n, n + 3]:
            shards = tt_saprot._shard_by_length(items, shards_n, key=aa_key)
            flat = [sid for sh in shards for sid, _ in sh]
            assert sorted(flat) == sorted(sid for sid, _ in items), \
                f"coverage broken n={n} shards={shards_n}"
            assert len(flat) == len(set(flat)), f"duplicate id n={n} shards={shards_n}"
            sizes = [len(sh) for sh in shards if sh]
            if sizes:
                assert max(sizes) - min(sizes) <= 1, \
                    f"unbalanced shards n={n} shards={shards_n}: {sizes}"
            fake = []
            for sh in shards:
                rows = [_FakeEmb(sid) for sid, _ in sh]
                rng.shuffle(rows)
                fake.append(rows)
            rng.shuffle(fake)
            out = tt_saprot._reassemble(items, fake)
            assert [e.id for e in out] == [sid for sid, _ in items], \
                f"order broken n={n} shards={shards_n}"
    print("[host] shard coverage / balance / reassembly-order: OK")


class _FakeEmb:
    def __init__(self, sid):
        self.id = sid


def run_shard_subprocess(device, seqs, model, workdir, idx, *, return_logits, batch_size):
    """Run one shard through the production _spawn_saprot_shard/_await_shard, serially."""
    h = tt_saprot._spawn_saprot_shard(idx, device, list(seqs.items()), workdir, model=model,
                                     fast=False, return_logits=return_logits, pool="mean",
                                     batch_size=batch_size)
    return tt_saprot._await_shard(*h)


def max_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m", choices=list(tt_saprot.CONFIGS))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--logits", action="store_true")
    args = ap.parse_args()

    check_host_logic()

    device = int(os.environ.get("TT_VISIBLE_DEVICES", "0"))
    seqs = make_seqs(args.n)
    items = list(seqs.items())
    workdir = tempfile.mkdtemp(prefix="tt-bio-saprot-multicard-parity-")
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
    shards = tt_saprot._shard_by_length(items, args.shards, key=lambda it: len(it[1][0]))
    shard_results = []
    for idx, sh in enumerate(shards):
        if not sh:
            continue
        r = run_shard_subprocess(device, dict(sh), args.model, workdir, idx,
                                 return_logits=args.logits, batch_size=1)
        shard_results.append(r)
        print(f"[shard {idx}] {len(sh)} seq -> {len(r)} embeddings "
              f"(aa-lens {sorted(len(v[0]) for _, v in sh)})")
    fan = tt_saprot._reassemble(items, shard_results)

    # Also exercise the production orchestrator itself (embed_multicard) end-to-end with a
    # single device — one shard, one subprocess, no same-card collision — so the real
    # mkdtemp/spawn/await/reassemble/cleanup path is covered, not just its primitives.
    orch = tt_saprot.embed_multicard(seqs, model=args.model, devices=[device],
                                     return_logits=args.logits, batch_size=1)
    assert [e.id for e in orch] == [sid for sid, _ in items], "orchestrator order != input"
    worst_o = 0.0
    for e in orch:
        r = ref[e.id]
        worst_o = max(worst_o, max_abs(e.per_residue, r.per_residue),
                      max_abs(e.pooled, r.pooled))
        if args.logits:
            worst_o = max(worst_o, max_abs(e.logits, r.logits))
    print(f"[orchestrator] embed_multicard(devices=[{device}]) order: OK  Δmax={worst_o:g}")

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
    bit_exact = (worst_pr == 0.0 and worst_pool == 0.0 and worst_o == 0.0
                 and (not args.logits or worst_lg == 0.0))
    if not bit_exact:
        print("FAIL: fanout is NOT bit-exact vs single-shot single-card")
        return 1
    print("PASS: multi-card fanout is BIT-EXACT vs single-shot single-card "
          "(split + subprocess transport + gather + reorder are lossless)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
