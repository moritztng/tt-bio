# The 0.897 s byte block, screened where its bytes actually are

`c10-fold-census` books 1,211.1 Mcycles / 0.8971 s above the traffic roof on `multiply_`, `add_` and
`layer_norm_w`, 843.2 Mc of it on `layer_norm_w`. This directory is the screen. Verdict NO-GO; the
full write-up is `state/c14-byte-deletion.md`.

Measured on qb1 node 0 (UMD card 3, p150a Blackhole, 11x10 cores) at a forced, during-sampled
1350 MHz, 155 samples, min = max = 1350.

## What is here

- `site_ledger.py` — per-op DRAM byte ledger for one captured `PairformerLayer`, deduped on buffer
  address. Reuses `perf/b2x_difflayer/itemize.py`, so the totals compare with `ROOF_RESIDUAL.md`.
- `ledger_qb1n0.json`, `captures/` — the ledger and the ttnn graph captures it reads, from one
  512 aa cdk2x2 fold on current main (`fc876c940`, ttnn 0.68.0).
- `lnbind.py`, `lnbind_qb1n0.json` — the 6-arm interleaved ladder on the largest executed
  `layer_norm_w` key, with a same-shape `ttnn.clone` as the known-answer control.
- `PREREG.json` — written before any device arm ran; `lnbind.py` refuses to start without it.
- `attrib_qb1n0.json` — the capture fold. Its wall is not a timing number: the
  harness syncs the device at every module bracket.

## The three numbers worth carrying

1. One deleted pass over the pair tensor at every trunk `PairformerLayer` call is **0.0418 s** at the
   in-fold 423.6 GB/s measured for the `add_` class (264 calls x 67.109 MB = 17.717 GB). A 0.3 s
   filter is therefore 7.2 deleted passes a layer, and one layer moves 100.61 Z = 6751.6 MB.
2. **All five z residual updates already come from L1** (ledger ops 23, 48, 74, 102, 194). The DRAM
   round trip a residual-accumulation lever would delete does not exist.
3. **Retargeting a pair-sized destination from DRAM to L1 deletes half the op's nominal bytes and
   returns 1.9 % of its time** (0.3812 -> 0.3741 ms, A/A floor 0.64 %). At that key `layer_norm_w`
   already runs at 96.5 % of a same-shape copy, and the copy itself reaches only 82.4 % of the
   442.9 GB/s roof the 843.2 Mc was booked against.
