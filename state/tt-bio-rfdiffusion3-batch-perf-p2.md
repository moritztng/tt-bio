# RFD3 multi-card designs/sec — MEASURED (p2)

Branch: `wk/tt-bio-rfdiffusion3-batch-perf-p2` (off merged p1 `01218f6f`).
Hardware: **qb2 QuietBox, 4× p300c (Blackhole)**. p1 baseline was **p150a (pc, 1 card)** —
same Blackhole arch, per-card compute identical, but p1 could not measure multi-card
(this is the real multi-card number p1 extrapolated).

Fixture: `IAI_protein.pdb`, contig `A1-10,20,A31-40` (I=40, L=419), `num_timesteps=200`,
`--from_pdb`, seed 42 (+i per design). H200 reference: **0.452 designs/sec** (rc-foundry
bf16 AMP, diffusion_batch_size=8).

## MEASURED designs/sec (not extrapolated)

| config | wall-clock | designs/sec | per-card s/design |
|---|---|---|---|
| 1 card, alone (bench step-count) | — | **0.0434** | 22.9 (115.1 ms/step) |
| 1 card, alone (4-design sequential) | 90.1s / 4 | 0.0444 | 22.5 |
| **--devices=2**, 2/card (M=4) | 52.70s | 0.0759 | 26.4 |
| **--devices=2**, 4/card (M=8) | 102.83s | **0.0778** | 25.7 |
| **--devices=4**, 2/card (M=8) | 169.22s | 0.0473 | 84.6 |
| **--devices=4**, 4/card (M=16) | 333.87s | **0.0479** | 83.5 |
| 4 cards, independent procs (diag) | 88.8s / 1 each | 0.0455 | 88.0 |

All numbers are cold-start wall-clock of `tt-bio design` (one subprocess per card, each
loads weights + compiles once, then runs its shard). The 2/card vs 4/card designs/sec
is stable within each device count (0.0759→0.0778, 0.0473→0.0479), so the per-worker
fixed cost (weight load + cold compile) DOES amortize — it is NOT the bound. The real
non-linearity is **card contention**: each card slows down super-linearly as more cards
run concurrently (22.5s/design alone → 25.7s at 2-card → 83.5s at 4-card).

## Parity PASS at --devices>1 (bit-exact, verified)

`cmp` of standalone single-card (card 0, in-process) vs fanout `--devices=0,1,2,3` outputs:
- `iai_0.cif` (seed 42): BIT-IDENTICAL
- `iai_1.cif` (seed 43): BIT-IDENTICAL
- `iai_2.cif` (seed 44): BIT-IDENTICAL
- `iai_3.cif` (seed 45): BIT-IDENTICAL

Cross-card same-seed (seed 42 run standalone on each of cards 0,1,2,3) vs fanout `iai_0`
(seed 42): all 4 BIT-IDENTICAL. So each device's output matches a standalone single-device
run with the same seed, regardless of which card ran it — the p1 parity claim holds.

## The real gap vs H200 (0.452 designs/sec)

- qb2 best (2 cards): **0.0778/s → 5.8× short of H200.**
- qb2 4 cards: 0.0479/s → 9.4× short (WORSE than 2 cards).
- qb2 1 card: 0.0434/s → 10.4× short.

**The p1 "~10 cards to match H200" extrapolation is wrong on a single shared-bus host.**
It assumed linear per-card scaling (each card adds 0.0434/s). The measured per-card
throughput COLLAPSES with concurrency (3.7× slower at 4-card), so adding cards beyond 2
on one QuietBox makes aggregate throughput WORSE, not better. The optimum on qb2 is 2 cards
(0.0778/s); 4 cards (0.0479/s) is barely above 1 card and below 2 cards.

**Cards needed by the REAL measured scaling:** linear scaling only holds across INDEPENDENT
hosts (1–2 cards each, no shared-bus contention). At 1 card per independent host
(0.0434/s each, no contention): 0.452 / 0.0434 ≈ **10–11 independent single-card hosts** to
match H200 — which numerically matches the p1 figure, but ONLY because each card must be on
its own host. On a single 4-card QuietBox the hard ceiling is 0.0778/s (2 cards), nowhere
near H200. The bound is host-bus / UMD-dispatch contention across cards on one host, not the
per-worker fixed cost p1 assumed.

## Bug fixes landed (required for the measurement)

1. `tt_bio/main.py` `design_cmd`: set `TT_MESH_GRAPH_DESC_PATH` for p300 boards on the
   single-device in-process path. The fanout path set it per shard; the in-process path
   (no `--devices`, or `--devices` with one card) opened the device without it and crashed
   on a p300 QuietBox (`Custom fabric mesh graph descriptor path must be specified for
   CUSTOM cluster type`). Same pattern the embed/gen/saprot commands already use.
2. `tt_bio/rfd3_design.py` `_design_out_path`: wrap `out_dir` in `Path()`. The fanout
   pickles `out_dir` as `str` for the shard subprocess, so `_run_design_jobs` received a
   `str` and `str / str` raised `TypeError`. Latent p1 bug (never hit on pc's 1 card); it
   broke ANY multi-device `tt-bio design --devices` run with `num_designs>1` or multiple
   specs. This is why p1's "wired and parity-correct by construction" fanout was never
   actually measured on multi-card.

## Bench harness (`scripts/rfd3_port/bench_designs_per_sec.py`)

Extended (not replaced): `--multi-device` mode shells out to `tt-bio design --devices` and
measures aggregate wall-clock designs/sec at 2/card and 4/card; `GOLDEN_DIR` is now
overridable via `RFD3_GOLDEN_DIR` (qb2 has no `~/.coworker/artifacts/rfd3-goldens/capture`;
weights auto-download to `~/.boltz/rfd3/weights` on first `tt-bio design`).

## Reproduce

```
# single-card warm baseline (step-count method)
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:... TT_MESH_GRAPH_DESC_PATH=<p150 mgd> \
  RFD3_GOLDEN_DIR=~/.boltz/rfd3/weights python3 scripts/rfd3_port/bench_designs_per_sec.py

# multi-device aggregate (drops TT_VISIBLE_DEVICES so the parent sees all cards)
TT_MESH_GRAPH_DESC_PATH=<p150 mgd> RFD3_GOLDEN_DIR=~/.boltz/rfd3/weights \
  python3 scripts/rfd3_port/bench_designs_per_sec.py --multi-device
```
