# p3-sdpa — Predictions, committed before the device was opened

Protenix-v2, trunk only, 298 aa (`examples/prot300.yaml`, CDK2 / 1HCL, 35-sequence MSA).
qb1 card 0, ttnn 0.67.4, branch `wk/protenix-trunk--p3-sdpa`. Conversion throughout is charter
§4.9 **x524** for the SDPA / qkv / head-split sites; I recount it in my own fold.

Every row says what would count as having been wrong. Nothing below was measured before it was
written; the git timestamp on this file is the evidence.

## Roofs I will take on THIS card (card 0, not P1's card 2 — not inherited)

| # | prediction | number | wrong if |
|---|---|---:|---|
| R1 | DRAM read roof (clone DRAM->L1, 128 MB) | 400 GB/s +/- 5 % | outside 380-420 |
| R2 | DRAM copy roof (clone DRAM->DRAM) | 410 GB/s +/- 5 % | outside 390-431 |
| R3 | matmul-writer write roof, L1 operands, DRAM out, K=32 | 200 GB/s +/- 8 % | outside 184-216 |
| R4 | compute roof, square 4096, HiFi4, DRAM out | 128 TFLOP/s +/- 8 % | outside 118-138 |
| R5 | L1->L1 copy roof at the split's 32-row chunk | 750 GB/s +/- 20 % | outside 600-900 |

## Deliverable 1 — what chunk 320 does to a finished structure

Reasoning behind D1a: recycling re-embeds and every block layer-norms, so a 2.30 % per-block
deviation cannot compound multiplicatively over 524 without saturating; but 48 blocks x 10 cycles
will amplify it well past one block.

| # | prediction | number | wrong if |
|---|---|---:|---|
| D1a | end-of-trunk pair track `z_trunk`, relative RMSD chunk 320 vs chunk 64 | 6-30 % | < 3 % or > 60 % |
| D1b | end-of-trunk single track `s_trunk`, relative RMSD | 1-10 % | < 0.3 % or > 25 % |
| D1c | all-atom RMSD between the two output structures, same seed, same draws | 0.3-2.0 A | < 0.1 A or > 5.0 A |
| D1d | CA-RMSD is at or below the all-atom figure | 0.7x-1.0x of D1c | above the all-atom figure |
| D1e | TM-score between the two structures | >= 0.95 | < 0.90 |
| D1f | lDDT between the two structures | >= 0.90 | < 0.85 |
| D1g | mean pLDDT of the two folds agrees | within 2.0 points | differ by > 5 points |
| D1h | placed against the repo's shipped bar, chunk 320 lands OUTSIDE it | outside | inside |

D1h names a bar I have to verify first. Two candidates are in the record and I predict they are
**different quantities and the record has conflated them**: `scripts/integration_envelope.py`
carries `ABS_FLOOR["kabsch_rmsd"] = 0.05` Angstrom with `DEFAULT_MARGIN = 0.50`, which is a
structural bar; the "0.0185-0.0217 band" that P1 called "a structural RMSD in angstroms" comes from
`state/perfwar-attention-block-fusion.md` and I predict it is a **dimensionless block-level
rmsd/std**, not Angstroms. Wrong if the 0.0185 figure turns out to be in Angstroms after all.

## Deliverable 2 — chunk 320 landed behind an argument, measured in a fold

| # | prediction | number | wrong if |
|---|---|---:|---|
| D2a | standalone SDPA probe, chunk 64, `[298,8,320,32]` + `[1,8,320,320]` DRAM mask | 2775 us/call +/- 12 % | outside 2442-3108 |
| D2b | standalone SDPA probe, chunk 320 | 1759 us/call +/- 12 % | outside 1548-1970 |
| D2c | probe-derived ms/fold at my own counted call number | 1065 ms/fold +/- 12 % | outside 937-1193 |
| D2d | `pf_stack` wall falls | 18.68 s -> 17.6 s, delta 0.98-1.15 s | delta < 0.70 s or > 1.40 s |
| D2e | whole-fold wall falls by the pf_stack delta divided by 480/524 | fold delta 1.05-1.25 s | fold delta < 0.6x the pf_stack delta |
| D2f | delivered, from the fold | 1000-1250 ms/fold | outside 700-1400 |
| D2g | the probe figure and the fold figure agree | within 20 % of each other | the fold delivers < half the probe |
| D2h | the in-code "PCC 0.9999 vs the 256 config" comment is optimistic | my PCC 64-vs-320 <= 0.9998 | I measure >= 0.9999 |

## Deliverable 3 — the L1-resident split

| # | prediction | number | wrong if |
|---|---|---:|---|
| D3a | `nlp_create_qkv_heads` L1->L1 vs DRAM->DRAM at a 32-row chunk reproduces P1 | 2.54x +/- 25 % | outside 1.9x-3.2x |
| D3b | the split is a pure index move, so an L1 output is bit-exact against a DRAM output | `torch.equal` True, 0 differing elements | one element differs |
| D3c | a 32-row chunk's L1 buffers allocate at the fold's own `[1,298,320,256]` class | 15.7 MB in + 15.7 MB out fits in 190 MB | the allocator refuses |
| D3d | measured as a PAIR (projection + split), the L1 arm beats the DRAM baseline by less than the 338-405 bound on the split alone, because the projection's own write moves too | net 200-500 ms/fold | net negative, or net > 700 ms/fold |

## The overlap I must not double-count

| # | prediction | number | wrong if |
|---|---|---:|---|
| D4 | `p3-align-widen`'s 42.3 ms/fold of unaligned-key-axis penalty inside SDPA @1629 SHRINKS under chunk 320, because 298-vs-320 stops costing across 5 k-chunks and costs inside one | 10-31 ms/fold survives | stays above 35 ms/fold, or inverts past -10 |

## Cores and overlap, on every arm (charter, both mandatory columns)

| # | prediction | number | wrong if |
|---|---|---:|---|
| U1 | chunk-320 SDPA core-equivalents from a grid ladder stay near chunk 64's 58.2 of 110, because 298x8 work units still swamp the grid at one q-chunk | 58 +/- 20 % | outside 46-70 |
| O1 | the chunk-320 arm is still `compute + comm`, not `max()`: the bias leg does not move | bias leg 1150 us +/- 15 % at chunk 320 | the bias leg falls by more than 25 % when the core leg does |
