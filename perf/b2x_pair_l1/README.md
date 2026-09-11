# perf/b2x_pair_l1 — can the Boltz-2 pair tensor stay in L1 at 512 aa?

Bet P3 of the `boltz2-2x` campaign. Answer: no. Full write-up in
`~/.coworker/state/b2x-pair-l1-residency.md`.

Measured on a p300c (qb2 card 2), 11x10 grid, ttnn 0.68.0, 2026-09-11. The usable L1 is
**1,461,760 B x 110 banks = 160.79 MB** — the allocator's figure, not
`get_max_worker_l1_unreserved_size()`'s 1,532,416 B, which is 4.8 % high.

The pair tensor is 67.11 MB at bf16 and the trimul's channel loop peaks at 603.99 MB, so nothing
containing `z` fits. The one sub-chain that does is the per-chunk tail — the two transformed
operands and their product — and only once the fused in-projection group narrows from the shipped
4 to 2. Residency then deletes 402.65 MB per trimul and returns 0.74-0.87 ms; the narrowing that
made it possible costs 2.32-2.58 ms. Net 1.15-1.17x slower than shipped, bit-exact.

The triangle matmul will not take a sharded operand at all: it runs `fuse_batch=False` because the
channel chunk is its batch axis, and ttnn answers *"Batch fusion is required when input A is
sharded"*.

| script | what it answers |
|---|---|
| `tail_ctl.py` | the controlled ladder — group x tail placement, plus the sharded-operand probe. **The measurement.** |
| `tail_refusal.py` | the verbatim L1 refusal at the shipped group width |
| `tail_ladder.py` | the first, confounded ladder; kept because it is the evidence for the confound |
| `tail_share.py` | bank-share sweep, contaminated by the `_dram_oom` L1/DRAM confusion; kept as its evidence |

Run them against your own worktree, pinned to one card:

    PYTHONPATH=$PWD TT_BIO_REBLOCK_L1_N_MAX=1024 TT_VISIBLE_DEVICES=<card> \
      python3 perf/b2x_pair_l1/tail_ctl.py perf/b2x_pair_l1/out.json

`TT_BIO_REBLOCK_L1_N_MAX` matters: without it the hand-written channel move is declined on an L1
destination above N=352 and the arm measures the resulting `ttnn.permute` swap instead of
residency — which reads as 1.26x slower rather than 1.06x faster.

`tt_bio/tenstorrent.py` carries the instrument these scripts drive, `_trimul_tail_memory_config`,
gated on `TT_BIO_TRIMUL_TAIL_L1` and off by default. It is inert with the flag off.
