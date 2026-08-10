# p3-chunk128 — predictions, committed before the device was opened

X2. Protenix-v2, trunk only, 298 aa. qb1 card 0, ttnn 0.67.4, branch `wk/protenix-trunk--p3-chunk128`
off `wk/protenix-trunk--p3-sdpa` (`dcb15bb2`). The question: **is there an SDPA triangle-attention
chunk above 64 that is inside the shipped structural envelope?** Chunk 320 is dead at 2.962 A
CA-RMSD against a 0.1136 A bound. Chunk 128 has never been measured for parity at all.

The bar is `scripts/integration_envelope.py`: CA-RMSD <= envelope x 1.5 + 0.05 A, instantiated for
protenix at **0.1136 / 0.1286 / 0.0500 A** in `state/gate-v0.5.0/parity/`. The port has already
spent 0.0491 A of the tightest one, so by the triangle inequality a chunk change has at most
**0.0645 A** of room before `protenix-prot-msa` could breach.

## The mechanism I am predicting from

Chunk 64 and chunk 128 are the SAME class of computation: a chunked online softmax with a running
max and a running sum, rescaled at every chunk boundary. Only the number of chunks over the padded
320 axis differs (5 vs 3). Chunk 320 is a DIFFERENT class: one chunk, no online rescale at all.

If deviation were set by which class you are in, chunk 128 would be close to chunk 64 and pass. My
model says it is not: the SDPA output is bf16, so ANY reduction-order change lands at bf16 rounding
(~4e-3 relative) whatever the chunk count, and the trunk then amplifies it through 524 c_z=256
block executions and 10 recycles. On that model the chunk count barely matters and every rung above
64 sits in the same order of magnitude as chunk 320. **So I predict the answer is no, and the
interesting number is how far from 320 chunk 128 actually lands.**

| # | prediction | number | wrong if |
|---|---|---:|---|
| C1 | chunks 96, 128, 160, 256 are all legal SDPA program configs on this build at the fold's padded 320 axis | 4 of 4 legal | any raises, in which case that rung is reported as illegal, not as slow |
| C2 | op-level screen (synthetic q/k/v at the fold's own [298, 8, 320, 32] shape) relative RMSD of the SDPA output vs chunk 64, rises monotonically with chunk size | 96 < 128 < 160 < 256 < 320 | any rung exceeds a larger rung by more than 50 % of the larger rung's figure |
| C3 | that op-level screen is small in absolute terms at chunk 128 | 1e-3 to 1e-2 relative | below 1e-4 or above 5e-2 |
| C4 | end-of-trunk `z_trunk` relative RMSD, chunk 128 vs 64 | 0.10-0.30 (chunk 320 gave 0.3073) | below 0.02 or above 0.45 |
| C5 | end-of-trunk `s_trunk` relative RMSD, chunk 128 vs 64 | 0.005-0.020 (chunk 320 gave 0.0153) | below 0.001 or above 0.05 |
| C6 | end-of-trunk `z_trunk` PCC, chunk 128 vs 64 | 0.95-0.99 | below 0.90 or above 0.999 |
| C7 | **CA-RMSD after Kabsch, chunk 128 vs chunk 64, full 298 aa fold** | **1.0-3.0 A** | below 0.3 A or above 5.0 A |
| C8 | all-atom RMSD after Kabsch tracks CA within a factor 1.2 | 1.0-1.2x the CA figure | the all-atom figure is below the CA figure |
| C9 | **chunk 128 is OUTSIDE the 0.1136 A bound** | outside, by more than 5x | inside the bound |
| C10 | chunk 96, the smallest rung above 64, is also outside | outside | inside the bound |
| C11 | **no rung above 64 is inside the envelope, so the lever is dead in all its forms** | 0 of 4 rungs pass | any rung passes |
| C12 | the op-level screen badly under-predicts the fold. Ratio of (fold CA-RMSD / typical CA-CA distance, ~3.8 A) to the screen's relative RMSD at the same rung | > 30x | the screen predicts the fold within 10x |
| C13 | TM-score, chunk 128 vs 64 | 0.90-0.97 (chunk 320 gave 0.9144) | below 0.85 or above 0.99 |
| C14 | lDDT, chunk 128 vs 64 | 0.85-0.95 (chunk 320 gave 0.8768) | below 0.80 or above 0.98 |
| C15 | **the same-chunk control**: two independent processes at chunk 128, same seed, same card | unaligned all-atom RMSD exactly 0.0 A, `torch.equal` True on `z_trunk` and `s_trunk` | one element differs, in which case nothing below is attributable to the chunk |
| C16 | SDPA probe at the fold's shape with the production bias, chunk 128 | 2000-2250 us/call (chunk 64 = 2765.5, chunk 320 = 1735.3 on this card) | outside 1850-2450 |
| C17 | chunk 128 speedup over chunk 64 on THIS card at 0.67.4, bias present. X5's 1.41x is a qb2 / 0.68.0 figure and I predict qb1 0.67.4 comes in LOWER because the bias re-read leg does not shrink | 1.25-1.40x | outside 1.15-1.55x |
| C18 | probe-derived ms/fold at my own counted call number | 550-800 ms/fold | outside 400-1000 |
| C19 | `pf_stack` wall delta, chunk 128 vs 64, full fold | 0.55-0.80 s (chunk 320 gave 1.052 s) | below 0.35 s or above 1.05 s |
| C20 | the fold delivers MORE than the probe, as it did for chunk 320 (+5.4 %) | fold/probe in 1.00-1.12 | the fold delivers less than 0.90x the probe |
| C21 | the SDPA core leg (bias absent) falls with chunk size while the bias leg does not move | bias leg within 5 % of the chunk-64 bias leg at every rung | a rung's bias leg moves more than 15 % against the chunk-64 bias leg |
| C22 | core-equivalents of the 110-core grid fall as the chunk grows, because chunk removes compute and no traffic (chunk 64 = 58.1, chunk 320 = 32.0) | chunk 128 in 40-52 | outside 32-58 |
| C23 | structural deviation is NOT saturated at a decorrelation plateau: a smaller chunk really does move the structure less | chunk 96 CA-RMSD < chunk 256 CA-RMSD | the rungs are scattered within 25 % of each other with no trend |
| C24 | the fold is bit-deterministic across processes at every rung tested, so one warm fold per rung is a sufficient structural sample | 0.0 A within every rung | any within-rung pair differs |

**The single prediction this pass is really about is C11.** If it holds, SDPA chunking is closed for
good and the org stops carrying it. If it loses, the largest lever in the ledger is alive and the
next question is which rung.
